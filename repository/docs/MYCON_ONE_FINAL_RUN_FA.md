# MyCON — یک مسیر اصلی پایان‌نامه (تثبیت ۱۴ آبان ۱۴۰۵)

> **فقط یک نوت‌بوک:** [MyCON_FINAL_ACCEPTANCE.ipynb](../MyCON_FINAL_ACCEPTANCE.ipynb) در PR #31.
> **مسیر منتخب:** `BACKEND = "colmap_gsplat"`. مسیر `longsplat` فقط برای مقایسه پژوهشی نگه‌داری شده است.

## انتخاب نهایی و دلیل پژوهشی

مرجع تجربی: [SpatialScene3D v10.5](https://github.com/Ziro21/SpatialScene3D).
این پروژه از **فیلم معمولی موبایل از محیط داخلی** با COLMAP و gsplat به مدل 3DGS رسیده و ۱۵ نمای hold-out با PSNR=28.93، SSIM=0.903 و LPIPS=0.058 منتشر کرده است. توسعه‌دهنده توضیح داده که به علت دشواری کامپایل افزونه‌های native در Colab، مسیر MASt3R-SLAM را با COLMAP عوض کرد. این نتایج مربوط به **ویدئوی نمونه خود آن‌ها** است، نه ویدئوی MyCON.

ما کد بدون لایسنسِ منتشرشده آن نوت‌بوک را کپی نکرده‌ایم. **روش پردازش موفق آن** را با مسیر خودمان تطبیق داده‌ایم و برای مرحله آموزش، **کد رسمی و مستقل Apache-2.0** از [gsplat v1.5.3](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3/examples) را در یک checkout مجزا اجرا می‌کنیم. این نسخه wheel از پیش‌کامپایل‌شده Python3.10/Torch2.4/CUDA12.1 دارد؛ **هرگز `gsplat/main` متغیر و وابسته به Torch جدیدتر را استفاده نکنید.**

## اتصال دقیق به پایان‌نامه MyCON

| کار پژوهشی | انجام دهنده در این PR | خروجی / شرط |
|---|---|---|
| برداشت واقعی | `MyCON_recorder` موجود | ZIP اصلی حدود ۹۶ ثانیه، pose/IMU/metadata با SHA256 |
| انتخاب فریم | `mycon_colmap_gsplat.py` | حدود ۱۹۲ فریم تمام مدت، ۲fps، ۶۴۰px، چرخش یک‌باره |
| تعیین دوربین و هندسه | COLMAP اصلی + SIFT sequential | مدل `sparse/0`، حداقل ۶۰ دوربین و ۶۵٪ ثبت؛ هندسه کاندید غیرمتریک |
| بازسازی بصری | gsplat v1.5.3 official `simple_trainer.py` | GPU واقعی T4، Gaussian checkpoint، PLY و ویدئوی نماهای جدید |
| آزمون کیفیت | معیارهای خود gsplat در ۱/۸ تصاویر ثبت‌شده | JSON واقعی `PSNR/SSIM/LPIPS`، بدون جعل |
| بررسی هندسی BIM | `bim_candidate/colmap_sparse_unscaled.ply` و `stage8_handoff.json` | **مقیاس دلخواه**؛ Stage8/9 و پیشرفت BIM تا ثبت مستقل و QA ممنوع |
| زمان‌بندی و داشبورد | Stage 9/11 موجود MyCON | پس از عبور Stage8 از کنترل متریک؛ فعلاً از مدل بصری خروجی پیشرفت جعل نمی‌شود |

دقت کن: Hold-out مدل gsplat روی تصاویر کنارگذاشته‌شده از آموزش است؛ دوربین آن تصاویر قبلاً در COLMAP ثبت شده، بنابراین این یک آزمون **کیفیت ظاهر** است، نه اعتبارسنجی مستقل هندسه یا وضعیت دوربین.

## فقط این روش را اجرا کن

1. [نوت‌بوک واحد](https://colab.research.google.com/github/sayyedalimrj/MyCon/blob/feature/mycon-single-final-acceptance-20261105/repository/MyCON_FINAL_ACCEPTANCE.ipynb) را باز کن. Runtime را GPU/T4 بگذار.
2. فقط در سلول اول: `BACKEND = "colmap_gsplat"` و `MODE = "preflight"`. ZIP اصلی به‌صورت خودکار تشخیص داده می‌شود؛ اگر چند ZIP بزرگ پیدا شد `CAPTURE_ZIP` را با مسیر واقعی وارد کن.
3. `preflight` **آموزش ندارد**؛ از ZIP اصلی استفاده می‌کند، COLMAP را اجرا و کیفیت ثبت دوربین را می‌سنجد، سپس محیط Python3.10/Torch2.4.1+cu121 و **کرنل واقعی gsplat Forward/Backward** را در محیط جدا بررسی می‌کند.
4. سلول وضعیت را بزن. فقط وقتی دقیقاً `PREFLIGHT_COLMAP_AND_GSPLAT_CUDA_PASSED_NO_TRAINING` آمد، می‌توانی `MODE = "all"` را با اجازه مصرف GPU اجرا کنی.
5. `all` از مدل دوربین موجود استفاده می‌کند، آموزش اولیه **۷۰۰۰ مرحله‌ای** را با gsplat رسمی آغاز می‌کند و checkpoint، PLY، ویدئوی نماهای جدید و JSON ارزیابی را بررسی می‌کند. ۷۰۰۰ مرحله «نقطه شروع قابل‌آزمایش» است، نه تضمین کیفیت پایان‌نامه.
6. اگر آموزش قطع شد، **همان پوشه مدل نیمه‌کاره را پاک یا overwrite نکن**. از لاگ‌ها علت را مشخص کن؛ هر آزمایش تازه با شناسه `ATTEMPT` مستقل انجام می‌شود.

## مسیر خروجی روی Drive

```text
MyDrive/MyCon_R4/research_colmap_gsplat/<archive-sha16>/<ATTEMPT>/
  input_contract.json                # هویت ثابت آزمایش
  frames_receipt.json                # SHA فریم‌ها و پارامترها
  colmap_receipt.json                # تعداد واقعی دوربین ثبت‌شده
  scene/images/                      # فریم‌های استخراج‌شده غیرمخرب
  scene/sparse/0/                    # مدل رسمی COLMAP
  bim_candidate/
      colmap_sparse_unscaled.ply     # هندسه SfM جدا از Gaussian PLY
      stage8_handoff.json            # ممنوعیت پیشرفت BIM تا کنترل متریک
  gsplat_result/
      ckpts/ckpt_*.pt                # پارامترهای آموزش
      ply/point_cloud_*.ply          # مدل نمایشی Gaussian
      videos/traj_*.mp4              # جابه‌جایی نما
      stats/val_step*.json           # PSNR/SSIM/LPIPS واقعی
  logs/pipeline.log
  logs/train.log
  status.json
```

## محدودیت‌های صریح

- **اجرای T4 روی داده خود MyCON هنوز تأیید نشده است.** تست GitHub فقط Python/Notebook/قراردادها را بدون GPU بررسی می‌کند. وابستگی fused_ssim هنوز به نصب/کامپایل native نیاز دارد؛ wheel رسمی gsplat یک بخش مهم از ریسک را حذف می‌کند، نه تمام آن را.
- COLMAP در محیط‌های کم‌بافت، پرتاب‌دار یا فیلم‌های سریع ممکن است کمتر از شرط ثبت دوربین را بگیرد. در آن حالت آموزش GPU عمداً شروع نمی‌شود. شواهد فعلی تضمین نمی‌کنند ضبط ۹۶ ثانیه‌ای تو به شرط ۶۵٪ می‌رسد.
- داده `pose.csv`/IMU اصلی تغییر نمی‌کند و نباید با دوربین‌های SfM مخلوط یا بدون اعتبارسنجی به مقیاس متریک تعبیر شود.
- بدون کنترل مستقل هندسی سایت/BIM، مدل Gaussian یا COLMAP Sparse مستقیماً پیشرفت عناصر یا ابعاد دقیق را ثابت نمی‌کند.
- کد NVIDIA LongSplat و محیط‌های قبلی در PR و Drive محفوظ‌اند؛ این baseline روش اصلی تجربی ماست تا اول یک مدل واقعی و قابل‌ارزیابی تولید شود.

**پیش‌نیاز پذیرش PR:** preflight واقعی T4 روی ZIP اصلی → آموزش کامل → مشاهده ویدئو/مدل و اندازه‌گیری واقعی QA → سپس جداگانه کنترل متریک Stage8 و پیشرفت Stage9/11. PR تا طی شدن این مراحل ادغام نشود.

## شرط قطعی پژوهشی: مقایسه BIM بر زیبایی رندر اولویت دارد

یک خروجی «موفق پایان‌نامه» هرگز صرفاً فایل `nvs_rgb.mp4`، `point_cloud.ply` یا PSNR خوب نیست. دو خروجی همزمان و مستقل نیاز داریم:

1. **شاخه نمایشی:** LongSplat اصلی NVIDIA + فیلم و مدل 3DGS + held-out QA، بدون ادعای هندسه متریک.
2. **شاخه هندسی:** عمق خام `train/ours_<N>/depths/<image_name>.npy` و `cameras_all_train.json` به ابرنقاط مستقل (یا COLMAP/MVS در مسیر پشتیبان)؛ داده هندسی باید سپس با نقشه IFC واقعی و اندازه‌های کارگاهی کالیبره شود.

مطابق `NVlabs/LongSplat/render.py` نسخه pin‌شده، depth خام واقعاً با `np.save(... view.image_name + '.npy')` ذخیره می‌شود. تبدیل هندسی MyCON اکنون از intrinsics اصلی `scene/cameras.py` شامل `cx=width/2`, `cy=height/2` استفاده می‌کند و عدم تطابق اندازه دوربین/عمق را خطا می‌داند.

**مراحل اجباری پذیرش Stage 8:**
- فایل **IFC واقعی همان پروژه** و دستگاه مختصات آن؛ نمونه IFC مصنوعی برای تست کد کافی است ولی مدرک پیشرفت واقعی نیست.
- **حداقل چهار نقطه متناظر مستقل و غیرهم‌صفحه** در ساختمان، و **دو نقطه دیگر برای آزمون جدا از fit**، با ابعاد واقعی برحسب متر. CSV طبق `metric_anchor_gate.py`.
- تخمین *Sim(3)* با مقیاس **یکسان در همه محورها**؛ ممنوعیت کشش غیرخطی صرفاً برای چسباندن ابرنقاط به BIM.
- گزارش خطای fit/hold-out و کنترل جداگانه هم‌خوانی سطوح، پوشش فضایی، پایداری چندنما، خطای ثبت و visibility هر عنصر IFC. پیش‌فرض تحمل anchor `0.05m` فقط یک حد هدف قابل تنظیم است، نه میزان دقت اثبات‌شده.
- فقط بعد از تأیید مستقل Stage 8، Stage 9 حق محاسبه `observed / not-observed / uncertain` در سطح عناصر IFC با شناسه GlobalId دارد. نبود نقاط در بخش پوشیده **به معنی ساخته‌نشدن قطعی عنصر نیست**.
- سپس Stage 11 پیشرفت عناصر قابل‌مشاهده را طبق mapping فعالیت‌های زمان‌بندی مقایسه می‌کند.

**وضعیت فعلی:** خود فایل IFC واقعی، نقاط کنترل مستقل و خروجی ثبت متریک روی داده کامل MyCON هنوز تأیید نشده‌اند. بنابراین هیچ گزارش خودکار «پیشرفت واقعی کارگاه» را موفق نمی‌نامیم؛ خروجی فعلی فقط research candidate است. PR تا آزمون GPU، BIM و سطوح میدانی تأیید نشود ادغام نمی‌شود.

**منابع مرتبط:** [LongSplat/render.py](https://github.com/NVlabs/LongSplat/blob/main/render.py)، [Google ARCore Pose](https://developers.google.com/ar/reference/java/com/google/ar/core/Pose.html)، [BIM-Align at ISARC 2026](https://www.iaarc.org/publications/2026_proceedings_of_the_43rd_isarc_singapore/bim_align-an_automatic_framework_for_bim_to_as_built_alignment_with_multi_view_imagery_and_3d_gaussian_splatting.html).
