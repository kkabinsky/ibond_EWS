# ThaiBMA Corporate Bond Early Warning System

ระบบเตือนภัยล่วงหน้าสำหรับหุ้นกู้ภาคเอกชน พัฒนาสำหรับโครงการของสมาคมตลาดตราสารหนี้ไทย (ThaiBMA) ทำงานบนแผงข้อมูล iBond ที่มีตัวแปร 33 ตัว

โปรแกรมตอบสองคำถามหลักรายบริษัท

1. **ตอนนี้อยู่ห่างจากเส้นเตือนภัยเท่าไร** เส้นเตือนภัยตั้งจากกำลังคนที่ตรวจได้จริง ไม่ได้ตั้งจากค่าความน่าจะเป็นสัมบูรณ์
2. **ตัวแปรตัวไหนต้องขยับไปเท่าไรจึงจะข้ามเส้น** วัดเป็นจำนวนส่วนเบี่ยงเบนมาตรฐาน ซึ่งแปลงกลับเป็นค่าทางบัญชีได้

---

## เริ่มใช้งาน

```bash
git clone <repo-url>
cd thaibma
pip install -r requirements.txt
python run.py app
```

**ไม่ต้องตั้งค่า path ใด ๆ** โปรแกรมหาตำแหน่งฐานข้อมูลเองอัตโนมัติ โดยไล่หาไฟล์ `cmdf_credit.db` จากโฟลเดอร์ของตัวเองขึ้นไปตามลำดับชั้น และถ้ายังไม่มีฐานข้อมูล การติดตั้งกับการ import จะยังทำงานได้ตามปกติ จะมีข้อความบอกก็ต่อเมื่อมีคำสั่งที่ต้องเปิดฐานข้อมูลจริง ๆ

### วิธีรันแต่ละโปรแกรม

ไฟล์ทั้งหมด import กันเองด้วยชื่อสั้น เช่น `import cmdf_tree_classify` ซึ่งเดิมใช้ได้เพราะทุกไฟล์อยู่โฟลเดอร์เดียวกัน พอจัดเข้าโฟลเดอร์ย่อยเพื่อให้อ่านง่าย การ import แบบนั้นจะพัง `run.py` แก้ปัญหานี้ด้วยการใส่ทุกโฟลเดอร์ลง `sys.path` ก่อนเรียกโปรแกรม โค้ดทุกไฟล์จึงไม่ต้องแก้เลย

```bash
python run.py --list                        # ดูรายชื่อโปรแกรมทั้งหมด
python run.py app                           # เปิดหน้าจอโปรแกรม
python run.py hyperbolic_boundary_panel     # โมเมนตัมและเส้นไฮเพอร์โบลา
python run.py firm_shock_panel              # การ shock รายบริษัท
python run.py pairwise_shock_pd --help      # อาร์กิวเมนต์ส่งผ่านได้ตามปกติ
```

### ข้อมูล

ฐานข้อมูลที่ใช้งานจริงมีขนาด 375 MB และมี 155 ตาราง ซึ่งส่วนใหญ่เป็นตารางปฏิบัติการ เช่น log การส่งอีเมล คิวแจ้งเตือน และผลลัพธ์ระหว่างทาง ทั้งหมดนั้นไม่ได้อยู่ในคลังนี้ และ GitHub ก็ไม่รับไฟล์เดี่ยวที่เกิน 100 MB อยู่แล้ว

สิ่งที่งานวิจัยอ่านจริงมีสามตาราง แนบมาเป็น CSV บีบอัดรวม **1.4 MB** ในโฟลเดอร์ `dataset/`

| ตาราง | แถว | เนื้อหา |
|---|---|---|
| `ibond_33features_panel` | 16,986 | แผงเดือน-บริษัท 293 ผู้ออก ตัวแปร 33 ตัว ช่วง 2007-11 ถึง 2026-08 |
| `ibond_issuer` | 677 | ทะเบียนผู้ออกตราสาร |
| `ibond_default_payment` | 50 | เหตุการณ์ไม่ชำระที่บันทึกไว้ พร้อมลิงก์ประกาศของ ThaiBMA |

สร้างฐานข้อมูลกลับคืนด้วยคำสั่งเดียว

```bash
python dataset/build_db.py
```

ได้ไฟล์ `cmdf_credit.db` ขนาด 7.7 MB ที่โปรแกรมหาเจอเอง ตรวจแล้วว่าให้ผลตรงกับฐานข้อมูลเต็มทุกตัวเลข ทั้งจำนวนผู้ออก 293 ราย สถานะ HIGH RISK 15 ราย และเส้นเตือนภัย 0.026881

แท็บใน GUI ที่อ่านตารางปฏิบัติการ เช่น ประวัติการส่งอีเมล จะไม่มีข้อมูลแสดง เพราะตารางเหล่านั้นไม่ได้แนบมา ส่วนสคริปต์วิเคราะห์ทั้งหมดไม่ได้ใช้ตารางพวกนั้น

> ข้อมูลในแผงมาจากสื่อ iBond ของ ThaiBMA ก่อนนำไปเผยแพร่ต่อ โปรดตรวจสอบเงื่อนไขสิทธิ์การใช้งานของท่านเอง

### รหัสผ่าน

รหัสผ่าน iBond และ SMTP อ่านจากตัวแปรสภาพแวดล้อมเท่านั้น ไม่มีการเก็บไว้ในโค้ด

| ตัวแปร | ใช้ทำอะไร |
|---|---|
| `THAIBMA_USER` / `THAIBMA_PASS` | เข้าระบบ iBond |
| `SMTP_USER` / `SMTP_PASS` | ส่งอีเมลแจ้งเตือน |
| `THAIBMA_DATA` | ไม่จำเป็น ใส่เฉพาะกรณีวางฐานข้อมูลไว้ที่อื่น |

> ตามเงื่อนไขการใช้งานของ ThaiBMA รหัสผู้ใช้และรหัสผ่านเป็นความลับ ห้ามเปิดเผยแก่ผู้อื่นโดยไม่ได้รับอนุญาตเป็นลายลักษณ์อักษร เจ้าของบัญชีต้องตั้งค่าตัวแปรเหล่านี้บนเครื่องของตนเอง

---

## โครงสร้างโฟลเดอร์

```
thaibma/
├── run.py               ตัวเรียกโปรแกรม จัดการ sys.path ให้เอง
├── thaibma_paths.py     หาตำแหน่งฐานข้อมูลอัตโนมัติ
├── dataset/             ข้อมูลสามตาราง (csv.gz) + build_db.py สร้าง DB กลับคืน
├── app/                   6 ไฟล์
├── app/legacy/            7 ไฟล์
├── ews/                  14 ไฟล์
├── models/               17 ไฟล์
├── analysis/shock/        6 ไฟล์
├── analysis/pca/          5 ไฟล์
├── analysis/importance/   4 ไฟล์
├── reanalysis/            3 ไฟล์
├── data/                 22 ไฟล์
├── reporting/             9 ไฟล์
├── bench/                 4 ไฟล์
├── integrations/          3 ไฟล์
├── tests/                11 ไฟล์
```

---

## คู่มือ: โปรแกรมแต่ละตัวทำอะไร (113 ไฟล์)

คำอธิบายในตารางดึงมาจาก docstring ของแต่ละไฟล์โดยตรง ช่องที่ว่างคือไฟล์ที่ยังไม่มี docstring

### `ระดับบนสุด` — ไฟล์ระดับบนสุด

| โปรแกรม | ทำอะไร |
|---|---|
| `run.py` | launcher that makes the sorted folders behave like the flat layout |
| `thaibma_paths.py` | one place that decides where the data lives |

### `app` — หน้าจอโปรแกรม (Flet) และงานเบื้องหลัง เช่น การเฝ้าระวังและอีเมลแจ้งเตือน

| โปรแกรม | ทำอะไร |
|---|---|
| `app.py` | ThaiBMA Credit Early Warning System — Top Button Bar Tab Navigation & Dual Approach Framework |
| `email_alert_engine.py` | email_alert_engine.py ================================================================================ Automated Email Notification & Daily… |
| `final_complete_sidebar.py` | Final Complete Election Management System with Left Sidebar Navigation ALL functionality from original gui_app_flet.py without dialogs |
| `monitor_service.py` | scheduled iBond monitoring with email alerts |
| `notify.py` | email + Telegram alerting for the CMDF Credit EWS |
| `setup_credentials.py` | one-shot, safe setup of your ThaiBMA / iBond credentials |

### `app/legacy` — โปรแกรมหน้าจอรุ่นเก่า เก็บไว้อ้างอิง ไม่ใช่ตัวที่ใช้งาน

| โปรแกรม | ทำอะไร |
|---|---|
| `app10.py` | ThaiBMA Credit Early Warning System — Top Button Bar Tab Navigation & Dual Approach Framework |
| `app11.py` | ThaiBMA Credit Early Warning System — Top Button Bar Tab Navigation & Dual Approach Framework |
| `app12.py` | ThaiBMA Credit Early Warning System — Top Button Bar Tab Navigation & Dual Approach Framework |
| `app2.py` | ThaiBMA Credit Early Warning System — Top Button Bar Tab Navigation & Dual Approach Framework |
| `app3.py` | ThaiBMA Credit Early Warning System — Top Button Bar Tab Navigation & Dual Approach Framework |
| `app4.py` | ThaiBMA Credit Early Warning System — Top Button Bar Tab Navigation & Dual Approach Framework |
| `fix_view_and_sorting.py` | fix_view_and_sorting.py ================================================================================ Rebuilds `firm_issuer_mapping` and… |

### `ews` — แกนของระบบเตือนภัย ค่า PD เส้นเตือนภัย โมเมนตัม และการ shock รายบริษัท

| โปรแกรม | ทำอะไร |
|---|---|
| `CMDF_threshold.py` |  |
| `CMDF_threshold_fast.py` |  |
| `baselines.py` | anomaly scoring of the factor panel with the eight baseline detectors requested in the ExpoGAF-AnoNet review… |
| `bond_ews.py` | Approach 1 (discrete-time survival hazard) applied directly to the iBond corporate-bond data |
| `bond_ews_xgb.py` | Corporate Bond Early Warning System (Approach 2 — Machine Learning XGBoost Survival Hazard) |
| `firm_shock_panel.py` | per-issuer shock and threshold diagnostics for the GUI |
| `hyperbolic_boundary_panel.py` | momentum and the hyperbolic decision boundary on the real iBond panel, as a command-line tool and as a panel inside app.py |
| `lead_metrics.py` | Shared lead-time definitions for the CMDF credit app |
| `machine_survior.py` | Logistic vs XGBoost as the Approach-1 hazard estimator |
| `pd_threshold_monitor.py` | derive monitoring thresholds on determinant pairs from the shock analysis, so an issuer can be checked against a stated PD ceiling |
| `realtime_ews.py` | live early-warning scoring and LEAD TIME for Thai bond issuers |
| `survival.py` | Survival EWS on the firm panel — the PD_3M / momentum / hyperbolic-boundary machinery (Cox-style discrete-time hazard) applied to the real (or… |
| `survivor2.py` | Approach 1 ONLY (Dynamic Survival Hazard Early-Warning System) ================================================================================… |
| `threshold_design_figure.py` | a worked example of setting a monitoring threshold the way pd_threshold_monitor.py recommends: anchored on review capacity, not on an absolute… |

### `models` — โมเดลและการเปรียบเทียบโมเดล ทั้งกลุ่มต้นไม้ เส้นอัตราผลตอบแทน และ Koopman

| โปรแกรม | ทำอะไร |
|---|---|
| `cmdf_ai_pipeline_report.py` | consolidated development / testing / evaluation report for every AI engine built in this folder against the real iBond data |
| `cmdf_approach2_compare.py` | the Approach-2 calibrated hazard pipeline run with four different base learners, compared on identical data |
| `cmdf_ensemble.py` | two more methods on top of the four tree ensembles: Soft-Vote the plain average of the four base predictions |
| `cmdf_feature_select.py` | does a small, model-chosen feature set do as well as all 33? Three feature sets are compared on IDENTICAL rows of the iBond panel: |
| `cmdf_gbm_compare.py` | CatBoost / XGBoost / LightGBM comparison for ln(PD)_12m |
| `cmdf_tree_classify.py` | Random Forest / XGBoost / CatBoost / LightGBM against the logistic Approach-1 baseline, on the 33-feature iBond panel |
| `cmdf_tree_models.py` | Random Forest / XGBoost / CatBoost / LightGBM for ln(PD)_12m, with every table and figure emitted ready to \input into result.tex |
| `compare_aopproach2.py` | Read-only command-line comparison of Approach 1 and Approach 2 results |
| `compare_ibond_33features_models.py` | compare_ibond_33features_models.py ================================================================================ Evaluates and compares the… |
| `compare_models.py` | Head-to-head comparison of the two Approach-1 hazard engines |
| `curve_ml.py` | Machine-learning forecasting of the yield-curve factors (Level, Slope, Curvature) extracted by yield_curve_dns.py |
| `curve_xgb.py` | XGBoost forecasts of the yield-curve factors (Level / Slope / Curvature) with a full set of diagnostic plots |
| `koopman.py` |  |
| `koopman_gaf.py` | Approach 2 extension for the CMDF bond project |
| `ml_factors.py` | alternative factor engines for the CMDF bond project |
| `paper_replication.py` | paper_replication.py Replication of: Wattanatorn, W |
| `yield_curve_dns.py` | yield_curve_dns.py Dynamic Nelson-Siegel (DNS) decomposition of the Thai government bond yield curve into LEVEL, SLOPE and CURVATURE -- the… |

### `analysis/shock` — การ shock ตัวแปรทีละคู่และทีละสามตัว พร้อมพื้นผิวความเสี่ยง

| โปรแกรม | ทำอะไร |
|---|---|
| `knn_cluster_shock.py` | forty pairwise views of the determinant cloud, before and after a shock, with the clusters marked and the displacement measured |
| `koopman_shock_spectrum.py` | Koopman eigenvalues of the 33-determinant dynamics on the unit circle, one panel per model, before and after a determinant shock |
| `pairwise_shock_pd.py` | shock the real determinants two at a time and measure what happens to the predicted default probability across the whole 33-determinant panel |
| `pd_surface_3d.py` | the same determinant pairs as knn_cluster_shock.py, redrawn as three-dimensional response surfaces with the default probability on the z axis |
| `roe_triple_figures.py` | one figure per three-determinant combination containing ROE, saved as a separate JPG named after its determinants |
| `triple_shock_pd.py` | extend the shock analysis from pairs to triples, rank every three-determinant combination by its effect on the default probability, and write the… |

### `analysis/pca` — PCA ของการ shock และการแปลงกลับ (inverse PCA)

| โปรแกรม | ทำอะไร |
|---|---|
| `pca_inverse_derivation.py` | the inverse PCA map for the 20-point simulation, worked through with explicit numbers |
| `pca_inverse_surface.py` | the inverse PCA map for two and for three determinants, drawn as response surfaces, on the same 20-issuer shock simulation |
| `pca_shock_20points.py` | a small, readable version of the shock experiment |
| `pca_shock_analysis.py` | PCA of the two leading determinants, the inverse map, and what a threshold-level shock does to the regression outcome |
| `pca_shock_simulation.py` | synthetic-data illustration of how a feature shock moves an issuer cluster in PCA coordinates, and what that does to y |

### `analysis/importance` — ความสำคัญของตัวแปร ทั้ง gain, SHAP และ permutation

| โปรแกรม | ทำอะไร |
|---|---|
| `build_33feature_correlation.py` | Build a 33-feature sliding-window correlation image for the iBond panel |
| `make_importance_default.py` | feature importance measured against the REAL default event, for every tree model |
| `run_gaf_feature_important.py` | Generate firm-level GAF panels for important bond features |
| `run_gaf_features.py` | Generate Gramian Angular Field (GAF) panels for each of the 34 bond predictors in `feature_bond.xlsx` |

### `reanalysis` — การประเมินผลรอบใหม่ตามข้อท้วงติงของผู้ประเมินบทความ

| โปรแกรม | ทำอะไร |
|---|---|
| `reanalysis_nested.py` | nested tuning (R1.3) and a dependence-aware Diebold-Mariano test (R1.5) |
| `reanalysis_oof.py` | the reanalysis demanded by the reviewer report |
| `reanalysis_rules.py` | the decision-boundary comparison the reviewers asked for (fix-list item 6), computed rather than asserted |

### `data` — การดึงข้อมูล การสร้างแผง 33 ตัวแปร และการดูแลฐานข้อมูล

| โปรแกรม | ทำอะไร |
|---|---|
| `add_firm_name_to_tables.py` | add_firm_name_to_tables.py ================================================================================ Adds the `firm_name` column (full… |
| `build_firm_mapping_and_view.py` | build_firm_mapping_and_view.py ================================================================================ Creates a dedicated SQLite mapping… |
| `build_ibond_33features.py` | build_ibond_33features.py ================================================================================ Merges iBond corporate bond panel data… |
| `build_ibond_33features_latest.py` | build_ibond_33features_latest.py ================================================================================ Calculates the latest 33-feature… |
| `check_2026_features.py` |  |
| `check_hyperbola_data.py` |  |
| `check_tables.py` |  |
| `download_bond.py` | download Thai CORPORATE BOND data from iBond |
| `download_bound.py` | download_bound.py One-click iBond/ThaiBMA yield-curve download orchestration |
| `export_ibond_excel.py` |  |
| `fetch_real.py` | Download REAL firm data (SET / US) from the internet and map it into the 33-feature ThaiBMA schema — for both the ML app and the survival EWS |
| `gen_data.py` | Synthetic ThaiBMA credit-risk dataset (33 features) -> Excel |
| `ibond_client.py` | authenticated download of the Thai government bond yield curve from ThaiBMA / iBond |
| `ibond_grpc.py` | gRPC-Web client for ThaiBMA iBond |
| `inspect_and_fix_all_sqlite_tables.py` | inspect_and_fix_all_sqlite_tables.py ================================================================================ Scans all tables in SQLite… |
| `inspect_corp_view.py` |  |
| `list_db_tables.py` |  |
| `load_bond.py` | Load 33 real features from the ThaiBMA bond database (Rev01_Database_final.dta) |
| `map_bond_symbols_to_tables.py` | map_bond_symbols_to_tables.py ================================================================================ Maps issuer codes to their real… |
| `scan.py` | one-shot headless credit scan, for schedulers (openclaw, Windows Task Scheduler, cron) |
| `show_bond_mapping.py` |  |
| `v_ibond_33features_panel.py` | v_ibond_33features_panel.py ================================================================================ CLI Executable script that queries… |

### `reporting` — การสร้างตาราง รูป และไฟล์ LaTeX สำหรับรายงาน

| โปรแกรม | ทำอะไร |
|---|---|
| `add_appendix_slides.py` |  |
| `build_presentation.py` |  |
| `fix_tex_amp.py` |  |
| `flatten_tex.py` | turn a report that is split across dozens of \input fragments into one self-contained .tex file |
| `get_appendix_bond_list.py` | get_appendix_bond_list.py ================================================================================ Extracts the latest corporate bond list… |
| `make_auc_f1_table.py` | one table holding AUC and F1 for every method built in this folder, each expressed as a percentage improvement over the Approach-1 baseline |
| `make_result_update4.py` | builds result_update4.tex: the cross-dataset comparison table requested for CatBoost and LightGBM, filled from the databases in this folder |
| `make_table10.py` | Table 10 filled in for every model, not just the two original rows |
| `make_table9_dm.py` | Table 9 for every method, plus a Diebold-Mariano test against the Approach-1 baseline |

### `bench` — การวัดเวลาและการรันชุดใหญ่

| โปรแกรม | ทำอะไร |
|---|---|
| `benchmark_all.py` | Approach 1 vs Approach 2 vs basic Deep-Learning models |
| `benchmark_ews_runtime.py` | Benchmark the calibrated observed-event pipelines used in the manuscript |
| `run_survivor_ews_33features.py` | run_survivor_ews_33features.py ================================================================================ Executes the Survivor-2 EWS Engine… |
| `run_survivor_ews_33features_xgb.py` | run_survivor_ews_33features_xgb.py ================================================================================ Executes Approach 2… |

### `integrations` — การเชื่อมต่อระบบภายนอก

| โปรแกรม | ทำอะไร |
|---|---|
| `_patch_connector.py` |  |
| `openclaw_connector.py` | Local OpenClaw integration and persistence for the ThaiBMA EWS |
| `openclaw_worker.py` | Deterministic jobs run by the local OpenClaw cron scheduler |

### `tests` — สคริปต์ตรวจสอบว่าส่วนต่าง ๆ ยังทำงาน

| โปรแกรม | ทำอะไร |
|---|---|
| `test_33feature_correlation.py` |  |
| `test_33features_charts.py` | test_33features_charts.py ================================================================================ Generates: |
| `test_app2_alerts.py` |  |
| `test_font.py` |  |
| `test_imp.py` |  |
| `test_mapping.py` |  |
| `test_real_leadtime.py` | Unit tests for the shared actionable and persistent lead metrics |
| `test_realtime_leadtime.py` |  |
| `test_survival_visual_pipeline.py` |  |
| `test_thai.py` |  |
| `test_xgb_calibration.py` | test_xgb_calibration.py ================================================================================ Tests Platt scaling probability… |

---

## ข้อควรทราบเกี่ยวกับผลลัพธ์

ตัวเลขต่อไปนี้มาจากการรันจริงบนแผง 16,986 แถว ผู้ออก 293 ราย และควรอ่านพร้อมข้อจำกัด

- **เหตุการณ์ผิดนัดมีเพียง 32 เดือนจากผู้ออก 8 ราย** คิดเป็น 0.19% ของแถวทั้งหมด ตัวเลขความแม่นยำใด ๆ บนฐานนี้ต้องอ่านอย่างระวัง
- **สถานะ HIGH RISK ไม่ได้แปลว่าจะผิดนัด** ที่กำลังคน 5% มี 15 รายอยู่เหนือเส้น แต่ทั้งฐานข้อมูลมีผู้ออกที่เคยเกิดเหตุการณ์จริงแค่ 8 ราย
- **ค่า Brier Skill Score ติดลบ** ค่า PD ที่ได้ใช้จัดอันดับได้ แต่ไม่ควรอ่านเป็นความน่าจะเป็นตามตัวอักษร
- **เส้นไฮเพอร์โบลาไม่ได้ช่วยเพิ่มความแม่นยำบนแผงนี้** MCC ดีขึ้นเพียง +0.0013 เทียบกับกฎที่ดูระดับ PD อย่างเดียว
- **โมเมนตัมแทบไม่มีข้อมูลที่ภาพตัดขวางปัจจุบัน** ผู้ออก 289 จาก 293 ราย มีโมเมนตัมเท่ากับ 1.0000 พอดี เพราะแผงคัดลอกค่างวดล่าสุดต่อไปข้างหน้า งบการเงินออกรายไตรมาสแต่แผงเป็นรายเดือน
- **`Policyrate` น่าจะทำหน้าที่เป็นตัวระบุช่วงเวลา ไม่ใช่ปัจจัยความเสี่ยง** เหตุการณ์ทั้ง 32 เดือนอยู่ในควอดแรนต์เดียวคือหนี้สูงบวกดอกเบี้ยต่ำ และเกิดในช่วงท้ายของข้อมูลทั้งหมด

## ความต้องการของระบบ

Python 3.10 ขึ้นไป แพ็กเกจตาม `requirements.txt` หน้าจอโปรแกรมใช้ Flet 0.84 ขึ้นไป
