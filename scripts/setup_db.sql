-- scripts/setup_db.sql
-- ─────────────────────────────────────────────────────────────────────────────
-- One-time PostgreSQL bootstrap for the ERC Analyzer.
-- Run as a superuser:  psql -U postgres -f scripts/setup_db.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Create the database (skip if it already exists)
SELECT 'CREATE DATABASE erc_analyzer'
WHERE NOT EXISTS (
    SELECT FROM pg_database WHERE datname = 'erc_analyzer'
)\gexec

-- Connect to the new database before enabling extensions
\connect erc_analyzer

-- 2. Enable pgvector (requires pgvector to be installed on the OS first)
CREATE EXTENSION IF NOT EXISTS vector;

-- 3. Verify the extension is active
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Done — the Python app (app/db.py) will create all tables on startup.
