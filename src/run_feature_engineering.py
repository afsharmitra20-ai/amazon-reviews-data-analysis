"""
Phase 2 - Step 3: Execution script for domain feature engineering.
Integrates with MusicalFeatureEngineer, generates feature parquet files,
and compiles the formal Feature Dictionary markdown deliverable.
"""

import os
import joblib
import pandas as pd
from feature_engineering import MusicalFeatureEngineer

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("reports", "phase2")


def generate_feature_dictionary(fe: MusicalFeatureEngineer, out_path: str):
    """Compiles the mandatory Phase 2 Feature Dictionary markdown deliverable."""
    content = f"""# شناسنامه و دیکشنری ویژگی‌های مهندسی‌شده (Feature Dictionary)
**درس تحلیل داده - پاییز ۱۴۰۵ | مسیر ب (Amazon Musical Instruments)**
**فاز دوم: مهندسی ویژگی و کشف الگو**

این مستند توضیح‌دهنده تمامی ویژگی‌های مهندسی‌شده با منطق دامنه‌ای حوزه ساز و موسیقی است. تمامی ویژگی‌ها پیش از مشاهده مجموعه Test و بدون هرگونه نشت از برچسب هدف (`rating`) یا داده‌های آینده تعریف شده‌اند.

---

### جدول مشخصات و منطق دامنه‌ای ویژگی‌ها

| نام ویژگی (Feature Name) | نوع داده | حوزه تحلیلی | منطق دامنه‌ای و ارزش افزوده تحلیلی (Domain Justification) |
| :--- | :--- | :--- | :--- |
| `caps_ratio` | Float | لحن و احساس | نسبت حروف بزرگ به طول کل متن؛ شاخص عصبانیت، خشم یا تأکید اغراق‌آمیز بر عیب/مزیت کالا. |
| `exclamation_count` | Int | ساختار متن | تعداد علامت تعجب (!) در نقد؛ بازتاب هیجان یا شدت احساسات کاربر. |
| `exclamation_density` | Float | لحن و احساس | چگالی علامت تعجب نسبت به طول متن؛ سنجش بار هیجانی مستقل از بلندی متن. |
| `question_count` | Int | ساختار متن | تعداد علامت سؤال (؟) در نقد؛ شاخص ابهام، نقص در تنظیمات ساز یا عدم کارایی مورد انتظار. |
| `question_density` | Float | لحن و احساس | چگالی علامت سؤال نسبت به طول نقد؛ شاخص شک و تردید خریدار. |
| `avg_word_length` | Float | استایل‌سنجی | میانگین طول حروف هر کلمه؛ تفکیک اصطلاحات پیچیده فنی نوازندگان از واژگان ساده مبتدیان. |
| `lexical_diversity` | Float | غنای واژگانی | شاخص Type-Token Ratio (تنوع واژگان)؛ تفکیک نقدهای تحلیلی عمیق از جملات کوتاه و کلیشه‌ای اسپم. |
| `audio_quality_mentions` | Int | دامنه‌ای صوتی | بسامد اصطلاحات اکوستیک و صدادهی (*tone, sound, buzz, hum, pitch, tuning, acoustic*). |
| `build_durability_mentions`| Int | متریال و ساخت | بسامد واژگان کیفیت ساخت، خرابی و استحکام (*sturdy, cheap, plastic, wood, broke, defect*). |
| `value_price_mentions` | Int | ارزش اقتصادی | بسامد واژگان ارزیابی قیمت و ارزش خرید (*price, worth, expensive, cost, bargain, value*). |
| `feature_bullets_count` | Int | کاتالوگ ساز | تعداد نکات فنی و ویژگی‌های ثبت‌شده در بروشور کالا در آمازون. |
| `has_user_images` | Binary | چندوجهی کاربر | آیا خریدار برای بررسی یا نشان دادن عیب ساز عکس آپلود کرده است؟ |
| `has_product_images` | Binary | چندوجهی کاتالوگ| وجود تصاویر رسمی در کاتالوگ فروشنده کالا. |
| `review_age_days` | Float | چرخه عمر زمانی | سن نقد (روز) نسبت به تاریخ مرجع لنگر آموزش (آموزش‌دیده بدون نشت: {fe.anchor_timestamp_}). |
| `price_to_word_ratio` | Float | تعامل چندوجهی | نسبت لگاریتم قیمت کالا به تعداد کلمات نقد؛ اتصال وجه ساخت‌یافته و وجه متنی کالا. |
| `price_imputed` | Float | قیمت‌گذاری | قیمت کالا با جایگزینی میانه آموزش جهت حذف اثر اقلام پرت لوتیرها. |
| `price_is_missing` | Binary | قیمت‌گذاری | نشانگر باینری غیبت قیمت کالا (حفظ سیگنال اقلام ناموجود یا سفارشی). |
| `price_scaled` | Float | قیمت‌گذاری | مقدار استانداردشده لگاریتم قیمت کالا (log1p). |
| `verified_purchase` | Binary | اعتبار خریدار | تأییدیه رسمی آمازون بر خرید قطعی کالا توسط نویسنده نظر. |

---

### تضمین علمی عدم نشت اطلاعات (Zero Leakage Assurance)
1. **زمان مرجع (Anchor Timestamp):** متغیر زمانی `review_age_days` بر اساس بیشترین زمان موجود در مجموعه Train ({fe.anchor_timestamp_}) مهار شده است و به ساعت سیستم یا آینده وابسته نیست.
2. **قرنطینه نشت هدف:** ویژگی‌های `average_rating`، `rating_number` و `helpful_vote` پیش از این مرحله حذف شده‌اند.
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[✓] Feature Dictionary compiled successfully to: {out_path}")


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("[*] Loading preprocessed Train, Val, and Test datasets...")
    train_path = os.path.join(PROCESSED_DIR, "train_preprocessed.parquet")
    val_path = os.path.join(PROCESSED_DIR, "val_preprocessed.parquet")
    test_path = os.path.join(PROCESSED_DIR, "test_preprocessed.parquet")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    # 1. Initialize Feature Engineer
    fe = MusicalFeatureEngineer()

    # 2. Fit strictly on Train
    print("[*] Fitting MusicalFeatureEngineer strictly on Train set...")
    fe.fit(df_train)
    print(f"    -> Learned Anchor Timestamp (Train Max): {fe.anchor_timestamp_}")

    # 3. Transform all splits
    print("[*] Transforming Train, Val, and Test splits with engineered features...")
    df_train_feat = fe.transform(df_train)
    df_val_feat = fe.transform(df_val)
    df_test_feat = fe.transform(df_test)

    # 4. Save feature datasets
    out_train = os.path.join(PROCESSED_DIR, "train_features.parquet")
    out_val = os.path.join(PROCESSED_DIR, "val_features.parquet")
    out_test = os.path.join(PROCESSED_DIR, "test_features.parquet")

    df_train_feat.to_parquet(out_train, index=False)
    df_val_feat.to_parquet(out_val, index=False)
    df_test_feat.to_parquet(out_test, index=False)
    print(f"[✓] Engineered feature datasets saved to: {PROCESSED_DIR}")

    # 5. Persist the transformer object
    model_path = os.path.join(MODELS_DIR, "feature_engineer.joblib")
    joblib.dump(fe, model_path)
    print(f"[✓] Persisted Feature Engineer to: {model_path}")

    # 6. Generate Mandatory Feature Dictionary Deliverable
    dict_path = os.path.join(REPORTS_DIR, "features_dictionary.md")
    generate_feature_dictionary(fe, dict_path)

    # 7. Print summary audit
    audit_path = os.path.join(REPORTS_DIR, "feature_engineering_audit.txt")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write("=== PHASE 2 - STEP 3: FEATURE ENGINEERING AUDIT ===\n\n")
        f.write(f"Anchor Timestamp (Train Max): {fe.anchor_timestamp_}\n\n")
        f.write("SAMPLE STATISTICAL SUMMARY OF ENGINEERED FEATURES (TRAIN SET):\n")
        summary_cols = [
            "lexical_diversity", "caps_ratio", "exclamation_density",
            "audio_quality_mentions", "build_durability_mentions", "value_price_mentions",
            "price_to_word_ratio", "review_age_days"
        ]
        stats = df_train_feat[summary_cols].describe().round(4).T
        f.write(stats.to_string())

    print(f"[✓] Audit written to: {audit_path}")
    print("\n" + open(audit_path, "r", encoding="utf-8").read())


if __name__ == "__main__":
    main()
