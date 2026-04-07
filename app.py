from flask import Flask, request, jsonify
import boto3
import uuid
import io
import re
import os
from mindee import BytesInput, ClientV2, OCRParameters, OCRResponse
from boto3.dynamodb.conditions import Key
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# 🔐 CONFIG
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "med-reminder-prescriptions-yash")

MINDEE_API_KEY = os.getenv("MINDEE_API_KEY")
MINDEE_OCR_MODEL_ID = os.getenv("MINDEE_OCR_MODEL_ID")
MEDICATIONS_TABLE_NAME = os.getenv("MEDICATIONS_TABLE_NAME", "medications")
USERS_TABLE_NAME = os.getenv("USERS_TABLE_NAME", "users")
PORT = int(os.getenv("PORT", "5000"))
DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"


# =========================
# AWS CLIENTS
# =========================

s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY or None,
    aws_secret_access_key=AWS_SECRET_KEY or None,
    region_name=AWS_REGION
)

dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=AWS_ACCESS_KEY or None,
    aws_secret_access_key=AWS_SECRET_KEY or None,
    region_name=AWS_REGION
)

table = dynamodb.Table(MEDICATIONS_TABLE_NAME)
users_table = dynamodb.Table(USERS_TABLE_NAME)

mindee_client = ClientV2(api_key=MINDEE_API_KEY or None)


def validate_required_config():
    missing = []
    if not BUCKET_NAME:
        missing.append("BUCKET_NAME")
    if not MINDEE_API_KEY:
        missing.append("MINDEE_API_KEY")
    if not MINDEE_OCR_MODEL_ID:
        missing.append("MINDEE_OCR_MODEL_ID")
    return missing

# =========================
# 🔐 AUTH
# =========================

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        existing_user = users_table.get_item(Key={"email": email}).get("Item")
        if existing_user:
            return jsonify({"error": "User already exists"}), 400

        user_id = str(uuid.uuid4())
        users_table.put_item(
            Item={
                "email": email,
                "user_id": user_id,
                "password_hash": generate_password_hash(password),
                "created_at": datetime.utcnow().isoformat()
            }
        )

        return jsonify({
            "message": "Signup successful",
            "user_id": user_id,
            "email": email
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get("email") or "").strip().lower()
        password = data.get("password") or ""

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        user = users_table.get_item(Key={"email": email}).get("Item")
        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"error": "Invalid email or password"}), 401

        return jsonify({
            "message": "Login successful",
            "user_id": user["user_id"],
            "email": user["email"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 🔥 OCR HELPERS
# =========================

def extract_medicines(text):
    medicines = []
    seen = set()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    shared_instruction_text = re.sub(r"\s+", " ", text)
    blocker_words = (
        "diagnosis|advice|take|with|for|idiopathic|heading|date|body|name|age|address|closing|"
        "signature|refill|label|medical|centre|assistant|prof|neurology|reg|number|mouth|"
        "day|days|week|weeks|month|months"
    )
    frequency_pattern = r"(?:bid|bd|tid|td|qd|od)"
    entry_pattern = re.compile(
        rf"(?:(?P<prefix_freq>{frequency_pattern})\s+)?"
        r"(?:(?:tab|tablet|cap|capsule|inj|syp)\.?\s+)?"
        rf"(?P<name>(?!(?:{blocker_words}|{frequency_pattern}|tabs?|tablets?|caps?|capsules?)\b)"
        rf"[A-Z][a-zA-Z0-9-]*(?:\s+(?!(?:{blocker_words}|{frequency_pattern}|tabs?|tablets?|caps?|capsules?)\b)[A-Z][a-zA-Z0-9-]*){{0,2}})\s+"
        r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|ml)\b"
        r"(?P<tail>.*?)"
        rf"(?=(?:(?:{frequency_pattern})\s+)?(?:(?:tab|tablet|cap|capsule|inj|syp)\.?\s+)?"
        rf"(?!(?:{blocker_words}|{frequency_pattern}|tabs?|tablets?|caps?|capsules?)\b)[A-Z][a-zA-Z0-9-]*(?:\s+(?!(?:{blocker_words}|{frequency_pattern}|tabs?|tablets?|caps?|capsules?)\b)[A-Z][a-zA-Z0-9-]*){{0,2}}\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml)\b|$)",
        re.IGNORECASE,
    )

    for line in lines:
        normalized_line = re.sub(r"\s+", " ", line).strip()
        matches = list(entry_pattern.finditer(normalized_line))

        for index, match in enumerate(matches):
            name_tokens = match.group("name").strip(" .:-").split()
            while len(name_tokens) > 1 and name_tokens[0] and name_tokens[0][0].islower():
                name_tokens = name_tokens[1:]
            if len(name_tokens) > 1 and len(name_tokens[0]) == 1:
                name_tokens = name_tokens[1:]
            name = " ".join(name_tokens)
            key = name.lower()
            if key in seen:
                continue

            if index + 1 < len(matches):
                next_match = matches[index + 1]
                next_gap = normalized_line[match.end("unit"):next_match.start()].strip()
                next_boundary = next_match.start("name")
                if (next_match.group("prefix_freq") or "") and not next_gap:
                    next_boundary = next_match.start()
            else:
                next_boundary = len(normalized_line)
            prefix_freq = ""
            raw_prefix_freq = match.group("prefix_freq") or ""
            if raw_prefix_freq:
                if index == 0:
                    prefix_freq = raw_prefix_freq
                else:
                    gap_before_match = normalized_line[matches[index - 1].end("unit"):match.start()].strip()
                    if not gap_before_match:
                        prefix_freq = raw_prefix_freq
            medicine_context = normalized_line[match.end("unit"):next_boundary].strip()
            medicine_context = f"{prefix_freq} {medicine_context}".strip()
            tablets_context = medicine_context if has_tablet_count(medicine_context) else shared_instruction_text
            frequency_context = medicine_context if has_frequency_info(medicine_context) else shared_instruction_text
            duration_context = medicine_context if has_duration_info(medicine_context) else shared_instruction_text
            duration_text, duration_days = get_duration_info(duration_context)

            seen.add(key)
            medicines.append(
                {
                    "name": name,
                    "dosage": f"{match.group('dose')}{match.group('unit')}",
                    "tablets": get_tablet_count(tablets_context),
                    "duration_text": duration_text,
                    "duration_days": duration_days,
                    "times": get_times(get_frequency(frequency_context)),
                }
            )

    return medicines

    


def get_frequency(text):
    lower_text = text.lower()

    if "three times daily" in lower_text or "thrice daily" in lower_text:
        return 3
    if "twice daily" in lower_text or "two times daily" in lower_text:
        return 2
    if "once daily" in lower_text:
        return 1
    if re.search(r"\b3\s+times\s+daily\b", lower_text):
        return 3
    if re.search(r"\b2\s+times\s+daily\b", lower_text):
        return 2
    if re.search(r"\b1\s+time\s+daily\b", lower_text):
        return 1
    if re.search(r"\b(tid|td)\b", lower_text):
        return 3
    if re.search(r"\b(bid|bd)\b", lower_text):
        return 2
    if re.search(r"\b(qd|od)\b", lower_text):
        return 1

    if "1-0-1" in text or "1- 0 - 1" in text:
        return 2
    if "1-0-0" in text:
        return 1
    if "0-0-1" in text:
        return 1
    return 1


def has_frequency_info(text):
    lower_text = text.lower()
    return bool(
        "three times daily" in lower_text
        or "thrice daily" in lower_text
        or "twice daily" in lower_text
        or "two times daily" in lower_text
        or "once daily" in lower_text
        or re.search(r"\b[123]\s+times?\s+daily\b", lower_text)
        or re.search(r"\b(tid|td|bid|bd|qd|od)\b", lower_text)
        or "1-0-1" in text
        or "1- 0 - 1" in text
        or "1-0-0" in text
        or "0-0-1" in text
    )


def get_tablet_count(text):
    lower_text = text.lower()

    digit_match = re.search(r"\b(\d+)\s*(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules)\b", lower_text)
    if digit_match:
        return int(digit_match.group(1))

    word_to_number = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
    }
    word_match = re.search(
        r"\b(one|two|three|four)\s+(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules)\b",
        lower_text,
    )
    if word_match:
        return word_to_number[word_match.group(1)]

    return 1


def has_tablet_count(text):
    lower_text = text.lower()
    return bool(
        re.search(r"\b\d+\s*(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules)\b", lower_text)
        or re.search(
            r"\b(one|two|three|four)\s+(?:tab|tabs|tablet|tablets|cap|caps|capsule|capsules)\b",
            lower_text,
        )
    )


def has_duration_info(text):
    lower_text = text.lower()
    return bool(
        re.search(r"\bfor\s+\d+\s+(day|days|week|weeks|month|months)\b", lower_text)
        or re.search(
            r"\bfor\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\b",
            lower_text,
        )
    )


def get_duration_info(text):
    lower_text = text.lower()
    word_to_number = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    unit_to_days = {
        "day": 1,
        "days": 1,
        "week": 7,
        "weeks": 7,
        "month": 30,
        "months": 30,
    }

    digit_match = re.search(r"\bfor\s+(\d+)\s+(day|days|week|weeks|month|months)\b", lower_text)
    if digit_match:
        value = int(digit_match.group(1))
        unit = digit_match.group(2)
        return f"{value} {unit}", value * unit_to_days[unit]

    word_match = re.search(
        r"\bfor\s+(one|two|three|four|five|six|seven|eight|nine|ten)\s+(day|days|week|weeks|month|months)\b",
        lower_text,
    )
    if word_match:
        word = word_match.group(1)
        unit = word_match.group(2)
        value = word_to_number[word]
        return f"{word} {unit}", value * unit_to_days[unit]

    return None, None


def get_times(freq):
    if freq == 1:
        return ["09:00"]
    if freq == 2:
        return ["09:00", "21:00"]
    if freq == 3:
        return ["09:00", "15:00", "21:00"]
    return ["09:00"]


def medicine_already_exists(user_id, name, dosage, tablets, duration_days, time):
    response = table.query(
        KeyConditionExpression=Key('user_id').eq(user_id)
    )
    items = response.get("Items", [])

    for item in items:
        if (
            item.get("name", "").lower() == name.lower()
            and item.get("dosage") == dosage
            and item.get("tablets", 1) == tablets
            and item.get("duration_days") == duration_days
            and item.get("time") == time
        ):
            return True

    return False


def process_prescription_bytes(user_id, file_bytes, filename, s3_key=None):
    missing = validate_required_config()
    if missing:
        raise ValueError(f"Missing required config: {', '.join(missing)}")

    input_doc = BytesInput(file_bytes, filename)
    result = mindee_client.enqueue_and_get_result(
        OCRResponse,
        input_doc,
        OCRParameters(MINDEE_OCR_MODEL_ID)
    )

    extracted_text = ""
    for page in result.inference.result.pages:
        if page.content:
            extracted_text += page.content + "\n"

    print("OCR TEXT:\n", extracted_text)

    medicines = extract_medicines(extracted_text)
    inserted = []
    skipped_duplicates = []

    for med in medicines:
        for t in med["times"]:
            if medicine_already_exists(
                user_id,
                med["name"],
                med["dosage"],
                med["tablets"],
                med["duration_days"],
                t
            ):
                skipped_duplicates.append({
                    "name": med["name"],
                    "dosage": med["dosage"],
                    "tablets": med["tablets"],
                    "duration_text": med["duration_text"],
                    "duration_days": med["duration_days"],
                    "time": t
                })
                continue

            item = {
                "user_id": user_id,
                "medicine_id": str(uuid.uuid4()),
                "name": med["name"],
                "dosage": med["dosage"],
                "tablets": med["tablets"],
                "duration_text": med["duration_text"],
                "duration_days": med["duration_days"],
                "time": t
            }
            if s3_key:
                item["s3_key"] = s3_key

            table.put_item(Item=item)
            inserted.append(item)

    return {
        "message": "Upload + OCR + DB success",
        "extracted_text": extracted_text,
        "medicines": inserted,
        "skipped_duplicates": skipped_duplicates
    }

# =========================
# 📤 PRESIGNED S3 UPLOAD
# =========================

@app.route('/generate-upload-url', methods=['POST'])
def generate_upload_url():
    try:
        missing = validate_required_config()
        if missing:
            return jsonify({"error": f"Missing required config: {', '.join(missing)}"}), 500

        data = request.get_json(silent=True) or {}
        user_id = (data.get("user_id") or "").strip()
        file_name = (data.get("file_name") or "prescription.jpg").strip()
        content_type = (data.get("content_type") or "image/jpeg").strip()

        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        safe_file_name = file_name.replace("\\", "_").replace("/", "_")
        file_key = f"uploads/{user_id}/{uuid.uuid4()}-{safe_file_name}"

        upload_url = s3.generate_presigned_url(
            ClientMethod='put_object',
            Params={
                'Bucket': BUCKET_NAME,
                'Key': file_key,
                'ContentType': content_type
            },
            ExpiresIn=3600
        )

        return jsonify({
            "uploadUrl": upload_url,
            "fileKey": file_key
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/process-prescription', methods=['POST'])
def process_prescription():
    try:
        missing = validate_required_config()
        if missing:
            return jsonify({"error": f"Missing required config: {', '.join(missing)}"}), 500

        data = request.get_json(silent=True) or {}
        user_id = (data.get("user_id") or "").strip()
        file_key = (data.get("fileKey") or data.get("file_key") or "").strip()

        if not user_id or not file_key:
            return jsonify({"error": "user_id and fileKey are required"}), 400

        s3_object = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
        file_bytes = s3_object["Body"].read()
        filename = file_key.split("/")[-1] or "prescription.jpg"

        result = process_prescription_bytes(user_id, file_bytes, filename, s3_key=file_key)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# 📤 UPLOAD + OCR + STORE
# =========================

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        missing = validate_required_config()
        if missing:
            return jsonify({"error": f"Missing required config: {', '.join(missing)}"}), 500

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "Empty file name"}), 400

        user_id = (request.form.get("user_id") or "").strip()
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        filename = str(uuid.uuid4()) + ".jpg"

        # ✅ READ FILE ONCE
        file_bytes = file.read()

        # ✅ Upload to S3
        s3.upload_fileobj(io.BytesIO(file_bytes), BUCKET_NAME, filename)
        result = process_prescription_bytes(user_id, file_bytes, filename, s3_key=filename)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# 💊 ADD MEDICINE
# =========================

@app.route('/add-medicine', methods=['POST'])
def add_medicine():
    data = request.get_json(silent=True) or {}

    required_fields = ["user_id", "name", "dosage", "time"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    if medicine_already_exists(
        data["user_id"],
        data["name"],
        data["dosage"],
        data.get("tablets", 1),
        data.get("duration_days"),
        data["time"]
    ):
        return jsonify({"error": "Duplicate medicine already exists for this user and time"}), 400

    medicine_id = str(uuid.uuid4())

    table.put_item(
        Item={
            "user_id": data["user_id"],
            "medicine_id": medicine_id,
            "name": data["name"],
            "dosage": data["dosage"],
            "tablets": data.get("tablets", 1),
            "duration_text": data.get("duration_text"),
            "duration_days": data.get("duration_days"),
            "time": data["time"]
        }
    )

    return jsonify({
        "message": "Medicine added",
        "medicine_id": medicine_id
    })

# =========================
# 📥 GET MEDICINES
# =========================

@app.route('/get-medicines/<user_id>', methods=['GET'])
def get_medicines(user_id):
    response = table.query(
        KeyConditionExpression=Key('user_id').eq(user_id)
    )

    return jsonify(response['Items'])

# =========================
# ✏️ UPDATE
# =========================

@app.route('/update-medicine', methods=['PUT'])
def update_medicine():
    data = request.get_json(silent=True) or {}

    required_fields = ["user_id", "medicine_id", "name", "dosage", "time"]
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    table.update_item(
        Key={
            "user_id": data["user_id"],
            "medicine_id": data["medicine_id"]
        },
        UpdateExpression="SET #n = :name, dosage = :dosage, tablets = :tablets, duration_text = :duration_text, duration_days = :duration_days, #t = :time",
        ExpressionAttributeNames={
            "#n": "name",
            "#t": "time"
        },
        ExpressionAttributeValues={
            ":name": data["name"],
            ":dosage": data["dosage"],
            ":tablets": data.get("tablets", 1),
            ":duration_text": data.get("duration_text"),
            ":duration_days": data.get("duration_days"),
            ":time": data["time"]
        }
    )

    return jsonify({"message": "Medicine updated"})

# =========================
# ❌ DELETE
# =========================

@app.route('/delete-medicine', methods=['DELETE'])
def delete_medicine():
    try:
        data = request.get_json(silent=True) or {}

        required_fields = ["user_id", "medicine_id"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

        table.delete_item(
            Key={
                "user_id": data["user_id"],
                "medicine_id": data["medicine_id"]
            }
        )

        return jsonify({"message": "Medicine deleted"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =========================
# 🌐 TEST UI
# =========================

@app.route('/')
def home():
    return '''
    <h2>Quick Auth</h2>
    <p>Use POST /signup and POST /login with JSON body: {"email":"you@example.com","password":"123456"}</p>
    <h2>Presigned Upload</h2>
    <p>Use POST /generate-upload-url with JSON: {"user_id":"...","file_name":"prescription.jpg","content_type":"image/jpeg"}</p>
    <p>Then upload the image from app directly to S3 using the returned uploadUrl, and call POST /process-prescription with JSON: {"user_id":"...","fileKey":"..."}</p>
    <h2>Upload Prescription</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="text" name="user_id" placeholder="Enter user_id"><br><br>
        <input type="file" name="file">
        <button type="submit">Upload</button>
    </form>
    '''

# =========================

if __name__ == '__main__':
    app.run(debug=DEBUG, host="0.0.0.0", port=PORT)
