> **Historical Archive Note (v0.3.1):** This initial specification draft (`2026-11-18`) pre-dates the v0.3.0 native engine rewrite. References to linking `llama.cpp` as the primary backend are superseded: as of v0.3.0+, QERLIO's runtime inference engine under `core/inference/` is implemented from scratch without `llama.cpp`, and current support counts (`1 tested, 183 experimental`) are authoritative in [`README.md`](../README.md) and [`registry/model_architectures.json`](../registry/model_architectures.json).

# QERLIO Runtime
## [مسودة تأسيسية مؤرشفة — 18 نوفمبر 2026] المواصفات البحثية والهندسية الأولية

**الجهة المالكة للمشروع:** ZAQERIS AI — Research Group  
**الاسم المقترح للمحرك:** QERLIO Runtime (كيرليو)  
**الهوية المقترحة:** `QERLIO by ZAQERIS AI`  
**نوع الوثيقة:** Product Requirements Document + System Architecture + Technical Design + Historical Draft (`2026-11-18`, superseded by `v0.3.0` / `v0.3.1` where noted)  
**الإصدار:** 1.0 — Historical Archive Draft (`2026-11-18`)  
**التاريخ:** 18 نوفمبر 2026  
**حالة الاسم:** اسم مرشّح مؤقت، غير معتمد قانونياً  
**حالة المواصفات:** مسودة تأسيسية أولية مؤرشفة؛ المرجع الحالي للحالة الفعلية للمشروع هو [`README.md`](../README.md) وسجل البُنى [`registry/model_architectures.json`](../registry/model_architectures.json) (`1 tested, 183 experimental`).

---

> **تنبيه مهم حول الاسم والحقوق:** الاسم التجاري أو اسم البرنامج لا تحميه حقوق المؤلف بالطريقة نفسها التي تحمي بها الشيفرة والوثائق؛ الحماية المعتادة لاسم المنتج تكون من خلال العلامات التجارية، وقد تنشأ حقوق أخرى بحسب البلد والاستخدام. أُجري بحث ويب أولي عن الاسم المطابق `QERLIO` ولم يظهر في النتائج العامة التي تمت مراجعتها منتج برمجي أو محرك ذكاء اصطناعي واضح بالاسم المطابق. هذا لا يثبت أن الاسم متاح، ولا يُعد بحثاً قانونياً شاملاً أو ضماناً لقبول تسجيله. قبل الإعلان أو شراء النطاقات أو طباعة الهوية، يلزم فحص العلامات المطابقة والمتشابهة صوتياً وبصرياً في الأسواق المستهدفة، وقواعد WIPO والسجل المحلي المصري وسجلات الدول المستهدفة، خصوصاً الفئات المتعلقة بالبرمجيات وخدمات التقنية. الاسم في هذه الوثيقة **مؤقت وقابل للاستبدال**.

> **تنبيه هندسي:** QERLIO ليس وعداً بأن كل نموذج سيعمل على كل هاتف، ولا بأن NPU كل شركة يمكن الوصول إليه من واجهة موحدة، ولا بأن نموذجاً ضخماً مثل 70B سيعمل عملياً على هاتف عادي. الوثيقة تضع معمارية قابلة للقياس والتدرّج. كل تسريع خاص بجهاز أو مورّد يجب إثباته على عتاد حقيقي وبإصدارات SDK مدعومة.

---

# 1. الملخص التنفيذي الأولي (مسودة 2026-11-18 المؤرشفة)

QERLIO Runtime هو محرك ومنظومة تشغيل محلية (Local AI Runtime Platform) هدفها تحميل نماذج الذكاء الاصطناعي وتشغيلها وإدارتها محلياً على أجهزة المستخدم.

تتكون المنظومة في التصميم المرجعي من:
- **Inference Runtime:** طبقة تنفيذ وتشغيل النماذج (نُفذت فعلياً من الصفر في `v0.3.0+` تحت `core/inference/`).
- **Backend System:** واجهة تجريد لمحركات ومسارات تنفيذ مستقبلية (لم يُربط `llama.cpp` أبداً).
- **Model System:** فهرسة النماذج وتحميلها والتحقق من ملفاتها (`SHA-256` وحدود GGUF).
- **Adaptive Scheduler:** جدولة وموازنة الذاكرة والجلسات.
- **SDK/API:** واجهات C ABI وPython SDK وبوابة HTTP محلية.
- **Local AI Services:** خدمات أدوات الوكلاء الآمنة في بيئة معزولة.

**ملاحظة تاريخية حول قرار التنفيذ (مُنسوخ ومُحدَّث في `v0.3.0`):** اقترحت مسودة `2026-11-18` الأولية عدم كتابة kernels من الصفر والنظر في استخدام `llama.cpp` كـ Backend مبدئي لملفات GGUF؛ **إلا أن هذا المقترح لم يُنفذ وأُلغي رسمياً في `v0.3.0`** عندما بُني محرك GGUF v2/v3 الأصلي لـ QERLIO من الصفر داخل `core/inference/` (Python + NumPy + مكتبة C11 SIMD GEMM في `native_gemm.c`) بدون أي اعتماد تشغيلي على `llama.cpp` أو `ggml`.
