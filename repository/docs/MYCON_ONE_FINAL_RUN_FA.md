# MyCON — راهنمای یک نوت‌بوک CPU→T4 پس از ممیزی ۱۴ آبان ۱۴۰۵

## ارتقای ۶ نوامبر ۲۰۲۶ — کنترل پوشش و مسیر LongSplat مستقل

**نسخهٔ جدید:** نوت‌بوک در شاخهٔ PR #31 یک آزمایش جدید به نام `mycon_dual_input_g1_v5_20261106` ایجاد می‌کند. این آزمایش هیچ checkpoint، آرشیو Recorder، pose.csv یا IMU قبلی را حذف یا بازنویسی نمی‌کند.

- روی **CPU و قبل از COLMAP** مرحلهٔ `longsplat-dataset` یک مجموعهٔ **مستقل ۸fps/512px** در `longsplat_video8fps_512/images/` تولید می‌کند. SHA تک‌تک تصاویر، SHA ویدئو و اطلاعات تغییر جهت در `dataset_receipt.json` ثبت می‌شوند؛ دادهٔ تولیدشده برای LongSplat فقط **ورودی** است و اجرای NVIDIA/MASt3R یا `train_custom.sh` خودکار آغاز نمی‌شود.
- پس از آن فریم‌های مستقل خط مبنای COLMAP با تنظیم ۲fps/640px ساخته می‌شوند. `derived_frame_quality_qa.json` شاخص‌های تاری، روشنایی، تفاوت ظاهری و کاندیدهای پیشنهادی را ثبت می‌کند؛ **تا این نسخه کاندیدها به‌جای تصاویر COLMAP اعمال نمی‌شوند** و تغییر پیکسلی جای پارالاکس یا pose بومی را نمی‌گیرد.
- `temporal_coverage_qa.json` تصاویر ثبت‌شدهٔ COLMAP را به هشت بازهٔ زمانی تقسیم می‌کند، فاصله‌های ثبت‌نشده را اندازه می‌گیرد و در صورت عدم پوشش کافی، **قبل از T4 توقف ایمن می‌کند**. گذر از این گیت نیز هنوز اثبات اتصال هندسی، مقیاس متری، loop closure یا صحت BIM نیست.
- اگر یک فریمِ مشتق‌شده یا receipt قدیمی مغایر باشد، روال بازسازی‌های قبلی بدون حذف اصل Recorder حفظ می‌شود. این نسخه عمداً به‌جای تکیه بر مجموعه آزمایش‌های قبلی، ATTEMPT جدید دارد.
- مراحل **LongSplat training، هندسه dense/DA3، اتصال IFC، ارزیابی واقعی PSNR/SSIM/LPIPS و Stage9/11** هنوز برای تأیید پایان‌نامه، اجرای واقعی و مدارک مستقل نیاز دارند و در این تغییر انجام‌شده تلقی نمی‌شوند.

## اصلاحات ممیزی سوم (بدون اجرای GPU)

- توقف نصب/دانلود بی‌خروجی پس از **۶۰ دقیقه** به‌جای سقف عمومی ۸ ساعته؛ پیش‌اجرای واقعی ۲۰ مرحله‌ای سقف **۲۰ دقیقه**، آموزش کامل سقف **۸ ساعت** و ارزیابی نهایی سقف **۲ ساعت** دارد. این سقف‌ها صرفاً محافظ زمان‌اند، نه تضمین تکمیل خودکار مراحل.
- در بازیابی CPU، اگر `colmap_sparse_unscaled.ply` وجود داشته ولی `colmap_receipt.json` کامل نشده باشد، کد آن را کورکورانه نمی‌پذیرد؛ یک PLY جدید از **زیرمدل منتخب** تولید و هش مقایسه می‌کند. اگر متفاوت باشد PLY قبلی محفوظ می‌ماند و مدل درست انتخاب می‌شود.
- `stage8_handoff.json` به **شناسه زیرمدل، SHA مدل binary و SHA ابرنقاط PLY** متصل است. رسید قدیمیِ مغایر حفظ و یک رسید جدید تولید می‌شود.
- `sparse_qa` پیش از T4 حداقل ۱۰۰ نقطه سه‌بعدی، مختصات finite، خطای بازفرافکنی finite و حدود هندسی را بررسی و ثبت می‌کند. **Registered image count** به‌تنهایی به‌عنوان اثبات هندسه پذیرفته نمی‌شود.



**فقط [نوت‌بوک رسمی همین PR](https://colab.research.google.com/github/sayyedalimrj/MyCon/blob/feature/mycon-single-final-acceptance-20261105/repository/MyCON_FINAL_ACCEPTANCE.ipynb) را استفاده کن.** ZIPهای v1/v2 قدیمی منسوخ‌اند. مبنای نوت‌بوک، **COLMAP + gsplat v1.5.3** برای baseline واقعی است؛ NVIDIA LongSplat به‌صورت مستقل برای مقایسه پژوهشی باقی می‌ماند و خودکار آموزش نمی‌بیند.

## اجرا با دو بار Run all

### روی Runtime = CPU

۱. فایل اصلی Recorder ZIP باید در Drive متصل باشد. کد فقط آرشیو را با SHA/CRC/manifest می‌خواند؛ هیچ فایل Recorder، pose.csv، IMU یا checkpoint قبلی پاک نمی‌شود. اگر تشخیص یکتای ZIP ممکن نباشد، مقدار `CAPTURE_ZIP` را با **مسیر دقیق همان فایل واقعی** پر کن.

۲. `Run all` بزن. بخش‌های `verify`، `longsplat-dataset` (دادهٔ مستقل NVIDIA بدون آموزش) و سپس `geometry` روی CPU اجرا می‌شوند. اگر FFmpeg وسط تولید فریم قطع شود، فریم ناقص **قرنطینه می‌شود، حذف نمی‌شود** و دفعه بعد بازیابی ممکن است. تمام فریم‌ها در manifest SHA256 ثبت می‌شوند؛ کد فقط به هش تصویر اول و آخر اکتفا نمی‌کند.

۳. COLMAP کل فیلم ۹۶ ثانیه‌ای را در ۲fps/640px دریافت می‌کند. اگر چند مدل sparse ساخته شود، مدل با بیشترین تصویر ثبت‌شده انتخاب می‌شود، نه لزوماً `sparse/0`. خروجی‌ها، رسیدها و hash کدها روی Drive می‌مانند. PLY موجود بدون رسید نهایی نیز در صورت معتبر بودن حفظ می‌شود و رسید آن بازسازی می‌شود.

۴. پیام `INTENTIONAL_CPU_STOP` **توقف عمدی** است. بعد از `CPU_GEOMETRY_COMPLETE_ON_GOOGLE_DRIVE` دستی Runtime را به **Tesla T4 GPU** تغییر بده.

### روی Runtime = T4 GPU

۵. دوباره **Run all از بالا** بزن. نوت‌بوک محیط محلی را از اول می‌سازد ولی **مراحل CPU را Skip می‌کند**؛ چک می‌کند تمام فریم‌ها، receiptها، source lock و مدل COLMAP با همان کد قبلی سازگار باشند. اگر وسط فاصله CPU/GPU شاخه GitHub تغییر کرده باشد، قبل از نصب سنگین با پیغام `MYCON_SOURCE_CHANGED_BETWEEN_CPU_GPU_OR_RESUME` متوقف می‌شود و داده‌ها سالم می‌مانند.

۶. `preflight` محیط مستقل Python3.10، CUDA12.1، Torch2.4.1+cu121، gsplat1.5.3+pt24cu121 را آماده می‌کند. **GCC12 قبل از نصب/کامپایل native CUDA** انتخاب می‌شود. wheel رسمی gsplat دقیقاً با SHA256 انتشار رسمی کنترل می‌شود؛ بسته‌های واقعی نصب‌شده در `pip_freeze` ثبت می‌شوند. CUDA واقعی، fused-SSIM و LPIPS امتحان می‌شوند.

۷. بعد روی **همان تصاویر واقعی** ۲۰ گام optimizer اجرا می‌شود و در صورت موفقیت `PREFLIGHT_REAL_VIDEO_GPU_SMOKE_PASSED_NO_FULL_TRAINING` نوشته می‌شود. رسید این گام در Drive ذخیره می‌شود؛ بعد از عوض‌شدن Runtime و عبور دوباره آزمون CUDA پایه، آن را بی‌جهت تکرار نمی‌کنیم.

۸. آموزش کامل پیش‌فرض **خاموش** است. فقط اگر خودت `TRAIN_CONFIRMATION="RUN_7000_STEPS_T4"` را در اولین سلول گذاشته باشی، فرمان کامل `all` بعد از preflight اجرا می‌شود.

## بازیابی GPU و صرفه‌جویی در اعتبار

کد جدید `mycon_gsplat_resume_patch.py` کپی مستقلی از trainer رسمی gsplat 1.5.3 می‌سازد. در اولین ۲۰۰ گام و سپس هر ۵۰۰ گام و پایان، وضعیت کامل Gaussian، optimizer، scheduler، densification و RNG روی Drive ذخیره می‌شود؛ هر snapshot پس از بستن فایل **SHA256 و رسید size** دارد.

اگر Colab بعد از یکی از snapshotهای سالم قطع شود، دفعه بعد همان ATTEMPT را با Runtime T4 و `Run all` ادامه بده. کد snapshotها را از جدید به قدیم با hash امتحان می‌کند و snapshot خراب را انتخاب نمی‌کند. اگر قبل از اولین snapshot قطع شود، فایل‌های مشتق‌شده قدیمی را قرنطینه می‌کند و مرحله آموزش در همان ATTEMPT از ابتدا آغاز می‌شود، **بدون حذف اصل داده**. زمان‌بندی دقیق هر batch پس از قطع ممکن است bitwise یکسان نباشد؛ این تداوم optimizer است، نه replay کاملاً بیت‌به‌بیت.

`run_logged` خروجی تدریجی را می‌بیند، در سکوت غیرعادی طولانی یا تجاوز از سقف زمانی اجرا را متوقف می‌کند و logها را نگه می‌دارد. تعداد Gaussian و حافظه استفاده‌شده هم کنترل دارند؛ این guardها ریسک OOM را کم می‌کنند ولی OOM لحظه‌ای را ناممکن نمی‌کنند.

## خروجی Canonical (رفع مغایرت PLY و ارزیابی)

gsplat اصلی مدل را در مرحله پایانی **پیش از optimizer.step** هم ذخیره می‌کند؛ snapshot نهایی بعد از به‌روزرسانی است. از آنجایی که این دو دقیقاً یک مدل نیستند، اکنون مدل ارزیابی‌شده و PLY نهایی هر دو از **یک tensor state** در `gsplat_result/canonical/` ساخته می‌شوند. SHA checkpoint، SHA Gaussian PLY، SHA معیارهای holdout و SHA ویدئو به یک `canonical_evaluation.json` متصل‌اند؛ هیچ خروجی قبلی overwrite نمی‌شود.

trainer اصلاح‌شده `normalize_world_space=False` دارد تا Gaussian PLY و COLMAP در دستگاه مختصات اولیه یکسان باشند؛ **مقیاس هنوز دلخواه است، نه متری**. صرف PSNR/SSIM/LPIPS مطلوب اعلام صحت BIM نیست. وضعیت `research_quality_accepted` عمداً خودکار true نمی‌شود.

## اتصال صحیح به Stage 8 و پایان‌نامه

فایل `bim_candidate/colmap_sparse_unscaled.ply` یک cloud **sparse** است، نه مدل اندازه‌گیری‌شده دیوار/ستون. به همین دلیل تبدیل مستقیم آن به «درصد پیشرفت» ممنوع است.

ابزار جداگانه `repository/scripts/mycon_colmap_bim_adapter.py` وقتی قابل اجراست که **IFC واقعی** و فایل CSV نقاط کنترل مستقل موجود باشند. حداقل چهار نقطه غیرهم‌صفحه برای برازش Sim(3) و دو نقطه holdout نیاز است. بعد از قبولی خطای متریک، فقط `colmap_sparse_aligned_m.ply` و `stage8_review_config.yaml` را برای Stage8 **مرور** آماده می‌کند؛ Stage9 و Stage11 خودکار آزاد نمی‌شوند. پوشش سطحی، دیده‌شدن عناصر، نام GlobalIdها و نگاشت زمان‌بندی باید با پروژه واقعی تأیید شوند.

نمونه فرمان (مسیرها را فقط با فایل‌های واقعی عوض کن):

```bash
python repository/scripts/mycon_colmap_bim_adapter.py \
  --colmap-ply /content/drive/MyDrive/MyCon_R4/research_colmap_gsplat/.../bim_candidate/colmap_sparse_unscaled.ply \
  --anchors-csv /content/drive/MyDrive/MyCon_R4/<real-anchors>.csv \
  --ifc /content/drive/MyDrive/MyCon_R4/<real-model>.ifc \
  --site-config repository/configs/site01.yaml \
  --out /content/drive/MyDrive/MyCon_R4/<new-metric-review>
```

در این گام استفاده از `plyfile`، `numpy` و `PyYAML` لازم است. برای مقایسه نهایی، بافت Gaussian به‌تنهایی جای هندسه سطحی متریک را نمی‌گیرد.

## حدود اعتبارسنجی

GitHub CI با سورس واقعی gsplat v1.5.3، تست‌های fixture/CPU و synthetic IFC کنترل می‌شود. **هیچ یک به‌معنی اجرای موفق GPU T4 واقعی کاربر، آموزش کامل ۷۰۰۰ مرحله، صحت عددی checkpoint روی CUDA یا قبولی نهایی Stage8/9/11 با IFC واقعی نیست.** این موارد صرفاً با اجرای کنترل‌شده روی داده واقعی قابل تأیید هستند. من GPU اجرا نکرده‌ام.
