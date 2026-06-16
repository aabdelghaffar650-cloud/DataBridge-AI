# ════════════════════════════════════════════════════════
#  DataBridge AI — Constants
# ════════════════════════════════════════════════════════
APP_NAME        = "DataBridge AI"
APP_VERSION     = "1.0.0"
APP_SUBTITLE    = "Universal Data Intelligence & Analysis Platform"

MAX_HISTORY        = 10
MAX_UPLOAD_SIZE_MB = 100
MAX_HISTORY_MEM_MB = 500

BOOL_MAP = {
    'yes': 'Yes', 'no': 'No', 'نعم': 'Yes', 'لا': 'No',
    'y': 'Yes', 'n': 'No', 'true': 'Yes', 'false': 'No',
    'صح': 'Yes', 'خطأ': 'No', '1': 'Yes', '0': 'No',
}

PII_PATTERNS = [
    'name', 'phone', 'mobile', 'email', 'mail', 'national', 'id', 'ssn',
    'اسم', 'هاتف', 'جوال', 'بريد', 'هوية', 'رقم قومي',
]

SEMANTIC_GROUPS = {
    'ID / Identifier':        ['id', 'code', 'no', 'num', 'number', 'ref', 'serial',
                                'كود', 'رقم', 'معرف', 'مسلسل'],
    'Name':                   ['name', 'fullname', 'client', 'patient', 'employee',
                                'person', 'staff', 'user',
                                'اسم', 'عميل', 'موظف', 'شخص', 'مريض'],
    'Product':                ['product', 'item', 'sku', 'goods', 'service', 'article',
                                'merchandise', 'brand', 'model', 'variant', 'prod',
                                'منتج', 'صنف', 'بضاعة', 'سلعة', 'خدمة', 'عنصر', 'موديل'],
    'Company / Organization': ['company', 'organization', 'org', 'business', 'firm',
                                'enterprise', 'vendor', 'supplier', 'partner',
                                'contractor', 'institution', 'agency', 'corp', 'inc',
                                'شركة', 'مؤسسة', 'منظمة', 'جهة', 'مورد', 'مقاول', 'بائع'],
    'Date':                   ['date', 'time', 'period', 'year', 'month', 'day',
                                'datetime', 'timestamp',
                                'تاريخ', 'وقت', 'فترة', 'سنة', 'شهر', 'يوم'],
    'Status / Category':      ['status', 'state', 'category', 'type', 'class',
                                'group', 'segment', 'tag', 'label',
                                'حالة', 'نوع', 'فئة', 'تصنيف', 'مجموعة', 'شريحة'],
    'Value / Amount':         ['value', 'amount', 'total', 'sum', 'price', 'cost',
                                'revenue', 'sales', 'budget', 'salary', 'fee', 'payment',
                                'قيمة', 'مبلغ', 'إجمالي', 'مجموع', 'سعر', 'تكلفة',
                                'إيراد', 'مبيعات', 'ميزانية', 'راتب', 'رسوم'],
    'Count / Quantity':       ['count', 'qty', 'quantity', 'units', 'total_count', 'freq',
                                'عدد', 'كمية', 'وحدات', 'تكرار'],
    'Percentage / Rate':      ['rate', 'ratio', 'percent', 'pct', 'percentage',
                                'share', 'proportion',
                                'نسبة', 'معدل', 'حصة'],
    'Region / Location':      ['region', 'area', 'zone', 'district', 'city', 'country',
                                'state', 'province', 'address', 'location', 'site',
                                'branch', 'office',
                                'منطقة', 'محافظة', 'مدينة', 'دولة', 'عنوان', 'فرع', 'موقع'],
    'Gender':                 ['gender', 'sex', 'الجنس', 'جنس'],
    'Age':                    ['age', 'العمر', 'سن'],
    'Phone / Contact':        ['phone', 'mobile', 'contact', 'tel', 'fax', 'whatsapp',
                                'هاتف', 'جوال', 'رقم هاتف', 'واتساب', 'فاكس'],
    'Email':                  ['email', 'mail', 'e-mail', 'بريد', 'إيميل'],
    'Notes / Description':    ['note', 'notes', 'desc', 'description', 'comment',
                                'remark', 'detail', 'info',
                                'ملاحظة', 'وصف', 'تعليق', 'تفاصيل', 'معلومات'],
    'Score / Result':         ['score', 'result', 'grade', 'rank', 'rating',
                                'performance', 'kpi',
                                'درجة', 'نتيجة', 'تقييم', 'أداء', 'ترتيب'],
    'Unknown':                [],
}
