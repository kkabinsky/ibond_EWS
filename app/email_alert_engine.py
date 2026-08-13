# -*- coding: utf-8 -*-
"""
email_alert_engine.py
================================================================================
Automated Email Notification & Daily Scheduler System for iBond EWS.

Manages recipient email subscriptions, daily delivery schedules (HH:MM),
SMTP credentials, and generates rich HTML daily risk summary emails for
defaulted corporate bond issuers and high-risk alerts.
"""
import os
import sqlite3
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
from thaibma_paths import DATA_ROOT  # data lives outside the repo

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DATA_ROOT, "cmdf_credit.db")

def init_email_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS email_alert_config (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipient_email TEXT NOT NULL UNIQUE,
        recipient_name TEXT,
        schedule_time TEXT DEFAULT '08:30',
        alert_threshold TEXT DEFAULT 'HIGH RISK',
        is_enabled INTEGER DEFAULT 1,
        smtp_host TEXT DEFAULT 'smtp.gmail.com',
        smtp_port INTEGER DEFAULT 587,
        smtp_user TEXT,
        smtp_pass TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    
    cur.execute("""
    CREATE TABLE IF NOT EXISTS email_alert_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        recipient_email TEXT,
        subject TEXT,
        default_count INTEGER,
        high_risk_count INTEGER,
        status TEXT,
        details TEXT
    );
    """)
    
    # Insert default admin recipient if empty
    cur.execute("SELECT COUNT(*) FROM email_alert_config;")
    if cur.fetchone()[0] == 0:
        cur.execute("""
        INSERT INTO email_alert_config (recipient_email, recipient_name, schedule_time, alert_threshold, is_enabled)
        VALUES ('risk_officer@thaibma.or.th', 'ThaiBMA Risk Analyst', '08:30', 'HIGH RISK', 1);
        """)
        
    conn.commit()
    conn.close()

def save_email_config(email, name="Analyst", schedule_time="08:30", threshold="HIGH RISK", enabled=1, db_path=DB_PATH):
    init_email_db(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO email_alert_config (recipient_email, recipient_name, schedule_time, alert_threshold, is_enabled, updated_at)
    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ON CONFLICT(recipient_email) DO UPDATE SET
        recipient_name=excluded.recipient_name,
        schedule_time=excluded.schedule_time,
        alert_threshold=excluded.alert_threshold,
        is_enabled=excluded.is_enabled,
        updated_at=CURRENT_TIMESTAMP;
    """, (email.strip(), name.strip(), schedule_time.strip(), threshold, enabled))
    conn.commit()
    conn.close()

def delete_email_config(config_id, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM email_alert_config WHERE id = ?;", (config_id,))
    conn.commit()
    conn.close()

def get_email_configs(db_path=DB_PATH):
    init_email_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM email_alert_config ORDER BY id DESC;", conn)
    conn.close()
    return df

def get_email_logs(db_path=DB_PATH):
    init_email_db(db_path)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM email_alert_logs ORDER BY id DESC LIMIT 50;", conn)
    conn.close()
    return df

def generate_daily_default_summary_html(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    
    # 1. Fetch latest snapshot
    try:
        df_snap = pd.read_sql_query("SELECT * FROM v_ibond_33features_latest;", conn)
    except Exception:
        df_snap = pd.DataFrame()
        
    # 2. Fetch default records
    try:
        df_def = pd.read_sql_query("SELECT * FROM ibond_default_payment ORDER BY payment_date DESC;", conn)
    except Exception:
        df_def = pd.DataFrame()

    conn.close()

    total_bonds = len(df_snap)
    high_risk_bonds = len(df_snap[df_snap["alert"] == "HIGH RISK"]) if "alert" in df_snap.columns else 0
    watch_bonds = len(df_snap[df_snap["alert"] == "WATCH"]) if "alert" in df_snap.columns else 0
    total_def_records = len(df_def)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # High Risk rows HTML
    high_risk_html = ""
    if not df_snap.empty and "alert" in df_snap.columns:
        hr_df = df_snap[df_snap["alert"] == "HIGH RISK"].head(15)
        for _, r in hr_df.iterrows():
            cname = r.get("company_name", "-")
            bsym = r.get("bond_symbol", "-")
            pd3m = r.get("PD_3M", "-")
            mom = r.get("Momentum", "-")
            de = r.get("DE", "-")
            roa = r.get("ROA", "-")
            high_risk_html += f"""
            <tr style="border-bottom: 1px solid #fee2e2; background-color: #fff5f5;">
                <td style="padding: 8px; font-weight: bold; color: #991b1b;">{cname}</td>
                <td style="padding: 8px; font-family: monospace;">{bsym}</td>
                <td style="padding: 8px; font-weight: bold; color: #dc2626;">{pd3m}</td>
                <td style="padding: 8px;">{mom}</td>
                <td style="padding: 8px;">{de}</td>
                <td style="padding: 8px;">{roa}%</td>
            </tr>
            """

    if not high_risk_html:
        high_risk_html = "<tr><td colspan='6' style='padding: 12px; text-align: center; color: #166534;'>No High Risk Alerts Currently Flagged</td></tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Daily iBond Credit Risk & Default Early Warning Summary</title>
    </head>
    <body style="font-family: Arial, sans-serif; background-color: #f8fafc; margin: 0; padding: 20px; color: #1e293b;">
        <div style="max-width: 680px; margin: 0 auto; background: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.08); border: 1px solid #e2e8f0;">
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%); padding: 24px; color: #ffffff;">
                <h2 style="margin: 0; font-size: 20px;">Thai Corporate Bond EWS Daily Alert Summary</h2>
                <p style="margin: 6px 0 0 0; font-size: 13px; color: #93c5fd;">Automated Risk Monitoring Report & Daily Default Audit · {now_str}</p>
            </div>
            
            <!-- Summary KPI Cards -->
            <div style="padding: 20px;">
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr>
                        <td style="padding: 12px; background: #f1f5f9; border-radius: 6px; text-align: center; width: 25%;">
                            <div style="font-size: 11px; color: #64748b;">TOTAL BONDS</div>
                            <div style="font-size: 22px; font-weight: bold; color: #0f172a;">{total_bonds}</div>
                        </td>
                        <td style="padding: 12px; background: #fef2f2; border-radius: 6px; text-align: center; width: 25%;">
                            <div style="font-size: 11px; color: #991b1b;">HIGH RISK</div>
                            <div style="font-size: 22px; font-weight: bold; color: #dc2626;">{high_risk_bonds}</div>
                        </td>
                        <td style="padding: 12px; background: #fefce8; border-radius: 6px; text-align: center; width: 25%;">
                            <div style="font-size: 11px; color: #854d0e;">WATCH LIST</div>
                            <div style="font-size: 22px; font-weight: bold; color: #ca8a04;">{watch_bonds}</div>
                        </td>
                        <td style="padding: 12px; background: #eff6ff; border-radius: 6px; text-align: center; width: 25%;">
                            <div style="font-size: 11px; color: #1e40af;">DEFAULT RECORDS</div>
                            <div style="font-size: 22px; font-weight: bold; color: #2563eb;">{total_def_records}</div>
                        </td>
                    </tr>
                </table>

                <h3 style="color: #991b1b; font-size: 15px; border-bottom: 2px solid #fee2e2; padding-bottom: 6px; margin-top: 24px;">
                    🚨 Flagged High Risk Corporate Bond Issuers (Hyperbolic Alarm Zone)
                </h3>
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background-color: #f8fafc; color: #475569; text-align: left;">
                            <th style="padding: 8px; border-bottom: 2px solid #e2e8f0;">Company Name</th>
                            <th style="padding: 8px; border-bottom: 2px solid #e2e8f0;">Bond Symbol</th>
                            <th style="padding: 8px; border-bottom: 2px solid #e2e8f0;">PD (3M)</th>
                            <th style="padding: 8px; border-bottom: 2px solid #e2e8f0;">Momentum</th>
                            <th style="padding: 8px; border-bottom: 2px solid #e2e8f0;">D/E Ratio</th>
                            <th style="padding: 8px; border-bottom: 2px solid #e2e8f0;">ROA (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {high_risk_html}
                    </tbody>
                </table>

                <!-- Footer -->
                <div style="margin-top: 30px; padding-top: 16px; border-top: 1px solid #e2e8f0; font-size: 11px; color: #94a3b8; text-align: center;">
                    This is an automated alert generated by Thai Corporate Bond EWS Engine (33 Features & Hyperbolic Hazard Boundary).<br>
                    Data Sources: ThaiBMA, SET, BOT, Refinitiv ESG · CMDF Research Project
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content, high_risk_bonds, total_def_records

def send_daily_email_alert(recipient_email, smtp_user=None, smtp_pass=None, smtp_host="smtp.gmail.com", smtp_port=587, db_path=DB_PATH):
    html_body, high_risk_cnt, def_cnt = generate_daily_default_summary_html(db_path)
    now_date = datetime.datetime.now().strftime("%Y-%m-%d")
    subject = f"[iBond EWS Alert] Daily Corporate Bond Risk Summary — {high_risk_cnt} High Risk Flagged ({now_date})"
    
    status = "SUCCESS"
    details = "Email dispatched successfully."
    
    # If credentials are not supplied, simulate clean delivery & record audit log
    if not smtp_user or not smtp_pass:
        details = "Simulated delivery (No SMTP password set). HTML Report generated successfully."
    else:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = recipient_email
            
            part = MIMEText(html_body, "html")
            msg.attach(part)
            
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, recipient_email, msg.as_string())
            server.quit()
        except Exception as e:
            status = "FAILED"
            details = f"SMTP Error: {e}"
            
    # Record in SQLite audit log
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO email_alert_logs (recipient_email, subject, default_count, high_risk_count, status, details)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (recipient_email, subject, def_cnt, high_risk_cnt, status, details))
    conn.commit()
    conn.close()
    
    return status, details, html_body

if __name__ == "__main__":
    init_email_db()
    st, det, _ = send_daily_email_alert("test_risk@thaibma.or.th")
    print(f"Test Email Dispatch Status: {st} | Details: {det}")
    df_c = get_email_configs()
    print("Configured Email Subscribers:")
    print(df_c.to_string(index=False))
