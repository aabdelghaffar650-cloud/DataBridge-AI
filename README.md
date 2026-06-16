# DataBridge AI

Run:

```bash
streamlit run app.py
```

## First-time login setup

DataBridge AI no longer ships with a default password.

On first run, the app shows a setup screen where you create the local admin username and password. The password is stored only as a salted PBKDF2-SHA256 hash in:

```text
.databridge/auth.json
```

You can change credentials later from:

```text
MANAGE → Settings
```

Managed deployments can still override credentials using environment variables:

- `DATABRIDGE_USERNAME`
- `DATABRIDGE_PASSWORD`
- or `DATABRIDGE_PASSWORD_HASH`

## Supported data sources

### File upload

- CSV
- Excel: `.xlsx`, `.xls`
- JSON: `.json`
- JSON Lines: `.jsonl`, `.ndjson`
- Parquet: `.parquet`
- SQLite database files: `.db`, `.sqlite`, `.sqlite3`

Uploaded files are checked by size, extension, and MIME/signature validation. Raw `.sql` dump execution is blocked for safety.

### Database Connector

Open:

```text
DATA → Data Sources → Database Connector
```

Use a SQLAlchemy URL and a read-only `SELECT` query.

Examples:

```text
sqlite:///C:/data/my_database.db
postgresql+psycopg2://user:password@host:5432/database
mysql+pymysql://user:password@host:3306/database
```

## Language

Language switch supports English and Arabic from the login screen and sidebar.
