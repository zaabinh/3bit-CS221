/**
 * ABSA Restaurant Demo · script.js
 * Luồng: nhập review → debounce gọi /analyze → hiện tag gợi ý → bấm Đăng → card xuất hiện
 */

const reviewInput = document.getElementById("review-input");
const authorInput = document.getElementById("author-input");
const charNum = document.getElementById("char-num");
const tagPreview = document.getElementById("tag-preview");
const suggestedTags = document.getElementById("suggested-tags");
const postBtn = document.getElementById("post-btn");
const analyzeStatus = document.getElementById("analyze-status");
const reviewFeed = document.getElementById("review-feed");
const feedEmpty = document.getElementById("feed-empty");
const totalCount = document.getElementById("total-count");
const toast = document.getElementById("toast");

const ASPECT_META = {
  food: { emoji: "🍜", label: "Food" },
  service: { emoji: "🧑‍🍳", label: "Service" },
  price: { emoji: "💰", label: "Price" },
  ambiance: { emoji: "🌿", label: "Ambiance" },
  miscellaneous: { emoji: "📌", label: "Miscellaneous" },
};

let currentTags = []; // tags từ model cho review hiện tại
let reviewCount = 0;
let analyzeTimer = null;
let isAnalyzing = false;

// ── Char counter + debounce analyze ─────────────────────────
reviewInput.addEventListener("input", () => {
  const len = reviewInput.value.length;
  charNum.textContent = len;

  // Reset
  currentTags = [];
  postBtn.disabled = true;
  tagPreview.classList.add("hidden");
  analyzeStatus.textContent = "";

  clearTimeout(analyzeTimer);

  if (len < 5) return;

  // Debounce 800ms — chờ user gõ xong mới gọi API
  analyzeStatus.textContent = "Đang phân tích...";
  analyzeTimer = setTimeout(() => analyzeText(reviewInput.value.trim()), 800);
});

// ── Gọi /analyze ────────────────────────────────────────────
async function analyzeText(text) {
  if (!text || isAnalyzing) return;
  isAnalyzing = true;

  try {
    const resp = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await resp.json();

    if (data.aspects && data.aspects.length > 0) {
      currentTags = data.aspects;
      renderTagPreview(currentTags);
      analyzeStatus.textContent = "";
      postBtn.disabled = false;
    } else {
      analyzeStatus.textContent = "Không tìm thấy aspect.";
    }
  } catch {
    analyzeStatus.textContent = "Không thể kết nối server.";
  } finally {
    isAnalyzing = false;
  }
}

// ── Hiển thị tag preview ─────────────────────────────────────
function renderTagPreview(tags) {
  suggestedTags.innerHTML = "";
  tags.forEach((asp) => {
    const meta = ASPECT_META[asp] || { emoji: "🔹", label: asp };
    const span = document.createElement("span");
    span.className = "tag";
    span.dataset.aspect = asp;
    span.textContent = `${meta.emoji} ${meta.label}`;
    suggestedTags.appendChild(span);
  });
  tagPreview.classList.remove("hidden");
}

// ── Đăng review ─────────────────────────────────────────────
function postReview() {
  const text = reviewInput.value.trim();
  const author = authorInput.value.trim() || "Khách ẩn danh";
  if (!text || currentTags.length === 0) return;

  // Tạo card
  addReviewCard(author, text, currentTags);

  // Reset form
  reviewInput.value = "";
  authorInput.value = "";
  charNum.textContent = "0";
  currentTags = [];
  postBtn.disabled = true;
  tagPreview.classList.add("hidden");
  analyzeStatus.textContent = "";

  // Toast
  showToast();
}

// ── Tạo review card ──────────────────────────────────────────
function addReviewCard(author, text, tags) {
  // Ẩn empty state
  feedEmpty && feedEmpty.remove();

  reviewCount++;
  totalCount.textContent = `${reviewCount} đánh giá`;

  const initials =
    author === "Khách ẩn danh"
      ? "👤"
      : author
          .split(" ")
          .map((w) => w[0])
          .slice(-2)
          .join("")
          .toUpperCase();

  const card = document.createElement("div");
  card.className = "review-card";
  card.innerHTML = `
    <div class="card-header">
      <div class="card-avatar">${initials}</div>
      <div class="card-meta">
        <div class="card-author">${escHtml(author)}</div>
        <div class="card-time">Vừa xong</div>
      </div>
    </div>
    <p class="card-body">${escHtml(text)}</p>
    <div class="card-tags">
      ${tags
        .map((asp) => {
          const m = ASPECT_META[asp] || { emoji: "🔹", label: asp };
          return `<span class="tag" data-aspect="${asp}">${m.emoji} ${m.label}</span>`;
        })
        .join("")}
    </div>
  `;

  // Chèn lên đầu feed
  reviewFeed.insertBefore(card, reviewFeed.firstChild);
}

// ── Toast ────────────────────────────────────────────────────
function showToast() {
  toast.classList.remove("hidden");
  requestAnimationFrame(() => {
    requestAnimationFrame(() => toast.classList.add("show"));
  });
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => toast.classList.add("hidden"), 300);
  }, 2200);
}

// ── Utility ─────────────────────────────────────────────────
function escHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
