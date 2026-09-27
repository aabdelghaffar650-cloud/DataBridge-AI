# ════════════════════════════════════════════════════════
#  DataBridge AI — Constants
# ════════════════════════════════════════════════════════
APP_NAME        = "DataBridge AI"
APP_VERSION     = "1.20.0"
APP_SUBTITLE    = "Universal Data Intelligence & Analysis Platform"

MAX_HISTORY                 = 20
MAX_UPLOAD_SIZE_MB          = 100
HISTORY_MEMORY_SNAPSHOT_MB  = 16
MAX_HISTORY_DISK_MB         = 2048
# Backward-compatible legacy name. History no longer truncates large datasets.
MAX_HISTORY_MEM_MB          = HISTORY_MEMORY_SNAPSHOT_MB

BOOL_MAP = {
    "yes": "Yes", "no": "No", "نعم": "Yes", "لا": "No",
    "y": "Yes", "n": "No", "true": "Yes", "false": "No",
    "صح": "Yes", "خطأ": "No", "1": "Yes", "0": "No",
}

PII_PATTERNS = [
    "name", "phone", "mobile", "email", "mail", "national", "id", "ssn",
    "اسم", "هاتف", "جوال", "بريد", "هوية", "رقم قومي",
]

# Stage 6 ML-facing semantic types. Order is used by the review UI.
ML_SEMANTIC_TYPES = (
    "Identifier",
    "Numeric Continuous",
    "Numeric Discrete",
    "Currency",
    "Percentage",
    "Categorical",
    "Ordinal",
    "Boolean",
    "Datetime",
    "Free Text",
    "Email",
    "Phone",
    "Unknown",
)

# Name hints are supporting evidence only. Content analysis remains authoritative.
# Exact token matching is used, so names such as "paid" no longer match "id".
SEMANTIC_GROUPS = {
    "Identifier": [
        "id", "identifier", "uuid", "guid", "key", "code", "ref", "reference",
        "serial", "number", "no", "num", "رقم", "معرف", "كود", "مرجع", "مسلسل",
    ],
    "Numeric Continuous": [
        "measurement", "measure", "temperature", "weight", "height", "distance",
        "duration", "score", "average", "mean", "قياس", "حرارة", "وزن", "طول", "مدة", "متوسط",
    ],
    "Numeric Discrete": [
        "count", "qty", "quantity", "units", "frequency", "visits", "عدد", "كمية", "وحدات", "تكرار",
    ],
    "Currency": [
        "amount", "price", "cost", "revenue", "sales", "salary", "budget", "fee",
        "payment", "income", "expense", "value", "total", "مبلغ", "سعر", "تكلفة",
        "إيراد", "ايراد", "مبيعات", "راتب", "ميزانية", "رسوم", "قيمة", "إجمالي", "اجمالي",
    ],
    "Percentage": [
        "percent", "percentage", "pct", "rate", "ratio", "share", "proportion",
        "نسبة", "معدل", "حصة",
    ],
    "Categorical": [
        "category", "type", "class", "group", "segment", "status", "state", "tag",
        "label", "kind", "فئة", "نوع", "تصنيف", "مجموعة", "شريحة", "حالة",
    ],
    "Ordinal": [
        "rank", "level", "priority", "grade", "rating", "stage", "tier", "severity",
        "رتبة", "مستوى", "أولوية", "اولوية", "درجة", "مرحلة",
    ],
    "Boolean": [
        "is", "has", "flag", "active", "enabled", "valid", "yes no", "bool", "boolean",
        "نشط", "فعال", "صحيح", "مؤشر",
    ],
    "Datetime": [
        "date", "datetime", "timestamp", "time", "year", "month", "day", "period",
        "تاريخ", "وقت", "سنة", "شهر", "يوم", "فترة",
    ],
    "Free Text": [
        "description", "notes", "comment", "remarks", "details", "text", "message",
        "وصف", "ملاحظات", "تعليق", "تفاصيل", "نص", "رسالة",
    ],
    "Email": ["email", "mail", "e mail", "e-mail", "بريد", "إيميل", "ايميل"],
    "Phone": [
        "phone", "mobile", "telephone", "tel", "whatsapp", "fax",
        "هاتف", "جوال", "واتساب", "فاكس",
    ],
    "Unknown": [],
}

# Business meaning is retained as a secondary profile dimension. It no longer
# drives the ML type by itself.
BUSINESS_ROLE_GROUPS = {
    "ID / Identifier": [
        "id", "code", "no", "num", "number", "ref", "serial", "uuid", "guid",
        "كود", "رقم", "معرف", "مسلسل",
    ],
    "Name": [
        "name", "fullname", "client", "patient", "employee", "person", "staff", "user",
        "اسم", "عميل", "موظف", "شخص", "مريض",
    ],
    "Product": [
        "product", "item", "sku", "goods", "service", "article", "merchandise", "brand",
        "model", "variant", "prod", "منتج", "صنف", "بضاعة", "سلعة", "خدمة", "عنصر", "موديل",
    ],
    "Company / Organization": [
        "company", "organization", "org", "business", "firm", "enterprise", "vendor",
        "supplier", "partner", "contractor", "institution", "agency", "corp", "inc",
        "شركة", "مؤسسة", "منظمة", "جهة", "مورد", "مقاول", "بائع",
    ],
    "Date": [
        "date", "time", "period", "year", "month", "day", "datetime", "timestamp",
        "تاريخ", "وقت", "فترة", "سنة", "شهر", "يوم",
    ],
    "Status / Category": [
        "status", "state", "category", "type", "class", "group", "segment", "tag", "label",
        "حالة", "نوع", "فئة", "تصنيف", "مجموعة", "شريحة",
    ],
    "Value / Amount": [
        "value", "amount", "total", "sum", "price", "cost", "revenue", "sales", "budget",
        "salary", "fee", "payment", "قيمة", "مبلغ", "إجمالي", "مجموع", "سعر", "تكلفة",
        "إيراد", "مبيعات", "ميزانية", "راتب", "رسوم",
    ],
    "Count / Quantity": [
        "count", "qty", "quantity", "units", "total count", "freq", "عدد", "كمية", "وحدات", "تكرار",
    ],
    "Percentage / Rate": [
        "rate", "ratio", "percent", "pct", "percentage", "share", "proportion", "نسبة", "معدل", "حصة",
    ],
    "Region / Location": [
        "region", "area", "zone", "district", "city", "country", "state", "province", "address",
        "location", "site", "branch", "office", "منطقة", "محافظة", "مدينة", "دولة", "عنوان", "فرع", "موقع",
    ],
    "Gender": ["gender", "sex", "الجنس", "جنس"],
    "Age": ["age", "العمر", "سن"],
    "Phone / Contact": [
        "phone", "mobile", "contact", "tel", "fax", "whatsapp", "هاتف", "جوال", "رقم هاتف", "واتساب", "فاكس",
    ],
    "Email": ["email", "mail", "e mail", "e-mail", "بريد", "إيميل", "ايميل"],
    "Notes / Description": [
        "note", "notes", "desc", "description", "comment", "remark", "detail", "info",
        "ملاحظة", "وصف", "تعليق", "تفاصيل", "معلومات",
    ],
    "Score / Result": [
        "score", "result", "grade", "rank", "rating", "performance", "kpi",
        "درجة", "نتيجة", "تقييم", "أداء", "ترتيب",
    ],
    "Unknown": [],
}
