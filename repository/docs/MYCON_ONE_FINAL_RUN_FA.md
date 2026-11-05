# MyCON — فقط یک نوت‌بوک نهایی؛ اجرای کم‌ریسک و کنترل‌شده

**شروع فقط از این لینک:** [MyCON_FINAL_ACCEPTANCE.ipynb](https://colab.research.google.com/github/sayyedalimrj/MyCon/blob/feature/mycon-single-final-acceptance-20261105/repository/MyCON_FINAL_ACCEPTANCE.ipynb)

**مرجع نسخه‌ها:** [قفل سازگاری Python/CUDA/GPU](./MYCON_RUNTIME_COMPATIBILITY_LOCK_20261105.json)

## انتخاب فعلی

برای گرفتن اولین مدل قابل اندازه‌گیری، **COLMAP + gsplat رسمی v1.5.3** مسیر پیش‌فرض است. روش بر بازسازی از فیلم داخلی موبایل در [SpatialScene3D v10.5](https://github.com/Ziro21/SpatialScene3D) تکیه دارد، اما کد نوت‌بوک آن پروژه را کپی نمی‌کند. آموزش مستقیماً توسط [simple_trainer.py رسمی gsplat](https://github.com/nerfstudio-project/gsplat/tree/v1.5.3/examples) انجام می‌شود. این روش کنترل ساده‌تری برای استفاده با T4 دارد.

NVIDIA LongSplat و تمام checkpointهای قبلی **محفوظ‌اند**؛ مسیر LongSplat برای پژوهش مقایسه‌ای باقی می‌ماند. نصب PyTorch3D در ترکیب قبلی Torch 2.5.1 و کد اصلی NVIDIA نیازمند تست واقعی سازگاری است و در نوت‌بوک پیش‌فرض بدون تأیید آغاز نمی‌شود. این حفظ NVIDIA به معنی تأیید آموزش آن نیست.

## ترتیب فقط همین چهار مرحله

1. **Colab با Runtime = CPU**: سلول شماره ۱ اتصال به Drive و GitHub را اجرا کن. `CAPTURE_ZIP` را فقط اگر سیستم ZIP اصلی را پیدا نکرد با **مسیر واقعی همان آرشیو ۹۶ ثانیه‌ای** مقداردهی کن. سلول شماره ۲ را برای `verify` اجرا کن؛ این مرحله بدون pip، CUDA و آموزش است و فرمت/CRC/SHA256 ZIP اصلی و فضای آزاد را بررسی می‌کند.
2. **همچنان CPU**: سلول شماره ۳ را اجرا کن؛ `ffmpeg` و `COLMAP` در صورت نیاز با apt نصب می‌شوند، فریم‌های ۲fps با عرض ۶۴۰ از کل فیلم ساخته می‌شوند، COLMAP باید حداقل ۶۰ فریم و ۶۵٪ را ثبت کند. مدل هندسی `scene/sparse/0` و ابرنقاط مستقل `bim_candidate/colmap_sparse_unscaled.ply` در Drive ذخیره می‌شوند. GPU مصرف نمی‌شود.
3. **Runtime را به T4 GPU تغییر بده**: **سلول اتصال شماره ۱ را دوباره اجرا کن** چون Colab محیط محلی را عوض می‌کند. سپس سلول شماره ۴ `preflight` را اجرا کن. مرحله GPU با SHA ZIP و رسید CPU ثابت می‌کند کار CPU از قبل انجام شده و **ویدئو را دوباره استخراج نمی‌کند**. پس از نصب محیط جداگانه Python 3.10/Torch2.4.1/CUDA12.1/gsplat1.5.3، *واقعاً* CUDA forward/backward، fused-SSIM، LPIPS و CLI رسمی gsplat را امتحان می‌کند. هنوز آموزش نمی‌کند.
4. **فقط اگر پیام زیر صادر شد**: `PREFLIGHT_COLMAP_AND_GSPLAT_CUDA_PASSED_NO_TRAINING`. در سلول شماره ۵ مقدار `TRAIN_CONFIRMATION` را به **`RUN_7000_STEPS_T4`** تغییر بده و اجرا کن. ۷۰۰۰ مرحله صرفاً اولین خط مبنای تجربی است؛ کیفیت نهایی را تضمین نمی‌کند. سلول شماره ۶ وضعیت را بدون آموزش نشان می‌دهد.

**هرگز Runtime را بین مرحله پیش‌اجرای T4 و آموزش تغییر نده؛** بسته‌های محیط محلی به Runtime تعلق دارند و بعد از تغییر آن باید دوباره بررسی شوند. هرگز از دستور `pip install -U torch` خارج از محیط جداشده استفاده نکن.

## قفل سازگاری T4

| جزء | نسخه مورد انتظار |
|---|---|
| GPU | NVIDIA Tesla T4، SM 7.5، تقریباً ۱۵ GiB VRAM |
| Python آموزش | 3.10.13 در micromamba مستقل؛ Python میزبان Colab دست‌نخورده |
| Torch / torchvision | 2.4.1+cu121 / 0.19.1+cu121 |
| CUDA nvcc | 12.1 در محیط مستقل |
| gsplat | 1.5.3+pt24cu121، wheel رسمی cp310 لینوکس، بدون کامپایل gsplat |
| TorchMetrics / Tyro | 1.4.3 / 0.8.14 |
| NumPy / Pillow | 1.26.4 / 10.4.0 |
| OpenCV headless | 4.10.0.84 |
| ابزار SfM | FFmpeg + COLMAP روی CPU؛ نسخه apt در Runtime CPU به‌صورت واقعی آزموده می‌شود |

فهرست دقیق SHA wheel و مراجع upstream در فایل JSON همراه است. خروجی‌های source، CPU geometry و checksum روی Drive‌اند؛ پکیج‌های Python با Runtime محلی عوض می‌شوند. شروع آموزش بدون receipt معتبر **مسدود** است.

## خروجی‌ها و معیار پذیرش

```text
MyDrive/MyCon_R4/research_colmap_gsplat/<SHA16>/<ATTEMPT>/
  verified_capture_cpu.json      # همان Recorder، SHA اصلی و همه sidecarهای اصلی
  frames_receipt.json            # فریم‌ها و HASH ورودی
  colmap_receipt.json            # دوربین‌های ثبت شده و درصد ثبت
  scene/images/                  # از ویدئوی کامل بدون تغییر به Recorder
  scene/sparse/0/                # COLMAP
  bim_candidate/                 # ابرنقاط مستقل SfM، هنوز غیرمتریک
  gsplat_result/ckpts/           # checkpoint واقعی 3DGS
  gsplat_result/ply/             # Gaussian PLY
  gsplat_result/videos/          # فیلم حرکت دوربین جدید
  gsplat_result/stats/           # معیارهای واقعی holdout PSNR/SSIM/LPIPS
  logs/                          # خطای واقعی ابزارهای بالادستی
  status.json                    # READ-ONLY از طریق سلول وضعیت
```

**هیچ فایل PLY صرفاً به دلیل وجود، آماده مقایسه BIM نیست.** برای تطبیق نهایی باید IFC واقعی و دستگاه مختصات آن، حداقل چهار نقطه کنترل مستقل غیرهم‌صفحه و دو نقطه holdout جداگانه، برازش یکنواخت Sim(3)، سطح‌بندی و کنترل خطای هندسی/مشاهده‌پذیری هر عنصر تهیه شود. بعد از پذیرش Stage8، Stage9 می‌تواند عناصر دارای شواهد را به پیشرفت برنامه زمانی Stage11 منتقل کند. نبود شواهد در قسمت پوشیده ساختمان نشانه ساخته‌نشدن نیست.

## محدودیت مهم

GitHub CI صرفاً تست‌های بدون GPU و قراردادها را اجرا می‌کند. **تا زمانی که GPU واقعی T4 در همین نسخه و روی ZIP واقعی موفق نشده باشد، تضمین کارکرد همهٔ درایورها یا کیفیت BIM نداریم.** اگر در مرحله‌ای متوقف شد، log همان مرحله را بررسی کن و آن را کورکورانه تکرار نکن. هیچ checkpoint یا Recorder قبلی را حذف نکن.
