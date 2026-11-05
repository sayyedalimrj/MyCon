# راهنمای قطعی اجرای MyCON — یک نوت‌بوک، دو Runtime، بازیابی ایمن

**فقط [MyCON_FINAL_ACCEPTANCE.ipynb](https://colab.research.google.com/github/sayyedalimrj/MyCon/blob/feature/mycon-single-final-acceptance-20261105/repository/MyCON_FINAL_ACCEPTANCE.ipynb) را اجرا کن.** فایل ZIP و نوت‌بوک‌های v1/v2 قدیمی منسوخ‌اند.

## Run all اول روی CPU

- به Drive متصل شو. `CAPTURE_ZIP` را فقط اگر تشخیص خودکار یکتا شکست خورد با مسیر دقیق **آرشیو اصلی ۹۶ ثانیه‌ای Recorder** پر کن.
- `verify`: ZIP/manifest/CRC/hash/فضای آزاد را بدون GPU کنترل می‌کند. اصل ZIP، pose.csv، IMU و checkpointها تغییر نمی‌کنند.
- `geometry`: FFmpeg کل فیلم را با ۲fps و عرض ۶۴۰px به حدود ۱۹۲ فریم تبدیل می‌کند؛ COLMAP در حالت CPU، SIFT sequential و SfM را اجرا می‌کند؛ حداقل **۶۰ تصویر و ۶۵٪ تصاویر** باید ثبت شوند. تغییر نام گزینه‌های SIFT بین نسخه‌های COLMAP قبل از اجرای سنگین بررسی می‌شود.
- مدل `scene/sparse/0`، فریم‌ها، ابرنقاط مستقل `bim_candidate/colmap_sparse_unscaled.ply` و رسیدهای SHA بر روی **Drive** قرار می‌گیرند.
- پس از خروجی `CPU_GEOMETRY_COMPLETE_ON_GOOGLE_DRIVE`، سلول عمداً با `INTENTIONAL_CPU_STOP` متوقف می‌شود. **خطای عمدی پایان CPU است؛ مجدد CPU Run all لازم نیست.**

## Run all دوم روی T4

- Runtime را از Colab دستی به **NVIDIA T4** تغییر بده؛ /content طبیعی است که پاک شود. در نوت‌بوک همان **Run all از اول** را بزن.
- سلول اول GitHub و Drive را وصل می‌کند. سلول دوم با دیدن T4 **همه مراحل CPU را Skip می‌کند**.
- `preflight`: SHA اصلی و رسیدهای Drive را می‌خواند و **بدون ویدئوی محلی و بدون اجرای دوباره FFmpeg/COLMAP** فقط تصاویر ثبت‌شده و sparse model را به /content سریع منتقل می‌کند.
- محیط مجزای `Python 3.10.13 / torch 2.4.1+cu121 / CUDA 12.1 / gsplat 1.5.3+pt24cu121` نصب و کرنل‌های واقعی، fused-SSIM، LPIPS، واردکردن trainer و سپس **۲۰ قدم بهینه‌سازی واقعی روی همان تصاویر کارگاه** آزمایش می‌شوند. این یک اجرای کوتاه GPU است و هزینه زمانی کم اما واقعی دارد؛ **آموزش کامل ۷۰۰۰ قدمی نیست**.
- موفقیت دقیق: `PREFLIGHT_REAL_VIDEO_GPU_SMOKE_PASSED_NO_FULL_TRAINING`.
- `TRAIN_CONFIRMATION=""` پیش‌فرض، مانع آموزش کامل می‌شود. برای اجازه صریح `TRAIN_CONFIRMATION="RUN_7000_STEPS_T4"` را در سلول اول بگذار و روی همان T4 دوباره Run all بزن؛ پس از پیش‌اجرا، `all` اجرا می‌شود. اگر این تأیید را از قبل در سلول اول گذاشته باشی، **همان Run all دوم** آموزش را بعد از preflight آغاز می‌کند.

## بازیابی بعد از قطع GPU

- کد رسمی `nerfstudio-project/gsplat` از tag `v1.5.3` بی‌تغییر clone می‌شود. کد `mycon_gsplat_resume_patch.py` در همان checkout یک **فایل جداگانه** `mycon_resumable_trainer.py` می‌سازد؛ trainer رسمی overwrite نمی‌شود.
- ذخیره immutable full-state `result/recovery/resume_step_*.pt` بعد از مرحله **۲۰۰** و سپس هر **۵۰۰** مرحله و پایان شامل Gaussian، optimizer، scheduler، densification strategy و RNG است. snapshot فقط پس از پایان قدم optimizer ساخته می‌شود.
- با قطع Runtime، همان ATTEMPT را نگه دار و روی Runtime جدید T4 دوباره Run all بزن؛ پیش‌اجرا وابستگی‌ها را دوباره بررسی می‌کند؛ `all` با تأیید صریح از آخرین snapshot قابل خواندن ادامه می‌دهد. قبل از قدم ۲۰۰ یک snapshot بازیابی کامل وجود ندارد؛ برای یک اجرای خیلی زود قطع‌شده، کد بدون تخریب فایل‌های قبلی متوقف می‌شود.
- `--ckpt` رسمی gsplat **فقط eval** است، نه resume optimizer. کد MyCON برای جلوگیری از ادعای غلط از آن برای resume استفاده نمی‌کند. اگر آخرین snapshot مرحله ۶۹۹۹ کامل است، `mycon_gsplat_finish.py` بدون آموزش مجدد مدل Gaussian PLY و رندر/معیارهای hold-out را کامل می‌کند.
- **تضمین bitwise replay نداریم**: ترتیب shuffle بین DataLoaderهای دو Runtime ممکن است تغییر کند؛ ولی وزن‌ها، optimizer، scheduler و densification از snapshot کامل برمی‌گردند.

## گزارش نهایی و معیار پایان‌نامه

در `MyDrive/MyCon_R4/research_colmap_gsplat/<archive_sha16>/<ATTEMPT>/`:
- `verified_capture_cpu.json`، `frames_receipt.json`، `colmap_receipt.json`؛
- `scene/images` و `scene/sparse/0`؛
- `bim_candidate/colmap_sparse_unscaled.ply` با هشدار **مقیاس دلخواه**؛
- `gsplat_result/recovery/` و `gsplat_result/ckpts/`؛
- `gsplat_result/ply/point_cloud_6999.ply`، `gsplat_result/videos/traj_6999.mp4` و `gsplat_result/stats/val_step6999.json`؛
- `logs/real_video_20_step_smoke.log` و `logs/train_or_resume.log` و `status.json`.

**مهم:** این pipeline خط مبنای بصری-هندسی COLMAP/gsplat است. روش رسمی NVIDIA LongSplat و کد depth-to-BIM در همان PR برای مقایسه پژوهشی محفوظ‌اند و اجرای خودکار GPU ندارند. هیچ PLY یا PSNR به‌تنهایی متریک/بیم نیست. برای اعلام درصد پیشرفت واقعی، فایل IFC واقعی، حداقل چهار نقطه کنترل مستقل غیرهم‌صفحه و نقاط hold-out، تبدیل Sim(3)، تحلیل خطای متریک و مشاهده‌پذیری عناصر لازم است؛ خروجی فعلی خودش نتیجه نهایی BIM تولید نمی‌کند.

**وضعیت اعتبارسنجی:** تمام تست‌های آفلاین CI شامل تطبیق source patch با gsplat رسمی v1.5.3 است، اما **آموزش کامل و سازگاری واقعی GPU Colab روی داده کاربر هنوز اجرا/تأیید نشده**. این محدودیت فقط با اجرای preflight و آموزش در Colab خودت روشن می‌شود؛ از قبل تضمین خطای صفر داده نمی‌شود.
