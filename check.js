
// ============================================================
// 状态
// ============================================================
let images = [];
let filteredImages = [];
let currentIndex = 0;
let currentLabels = {};
let currentCustomTags = [];
let labelConfig = {};
let currentFilter = 'all';
let autoLabelAvailable = false;
let roleNamesData = {};
let selectedRoles = [];
let selectedTagFilters = []; // 选中的标签筛选条件
let lastSelectedImageBeforeFilter = null; // 记住筛选前选中的图片

// ============================================================
// 初始化
// ============================================================
async function init() {
    await Promise.all([loadLabelConfig(), loadImages(), loadStats(), loadRoleNames()]);
    renderImageList();
    if (filteredImages.length > 0) {
        await selectImage(0);
    }
    checkAutoLabelSupport();
}

async function loadRoleNames() {
    try {
        const res = await fetch('/api/role-names');
        roleNamesData = await res.json();
    } catch {
        roleNamesData = {};
    }
}

async function loadLabelConfig() {
    const res = await fetch('/api/label-config');
    labelConfig = await res.json();
}

async function loadImages() {
    const res = await fetch('/api/images');
    images = await res.json();
    applyFilter();
}

async function loadStats() {
    const res = await fetch('/api/stats');
    const s = await res.json();
    document.getElementById('stat-total').textContent = s.total;
    document.getElementById('stat-done').textContent = s.annotated;
    document.getElementById('stat-auto').textContent = s.auto_labeled;
    document.getElementById('stat-verified').textContent = s.verified;
    document.getElementById('progress-bar').style.width = s.progress + '%';
    document.getElementById('progress-text').textContent = s.progress + '%';
}

async function checkAutoLabelSupport() {
    try {
        const res = await fetch('/api/stats');
        const s = await res.json();
        // Check if auto-label is available by trying the button state
        autoLabelAvailable = true;
    } catch {
        autoLabelAvailable = false;
    }
}

// ============================================================
// 过滤与搜索
// ============================================================
function applyFilter() {
    const search = document.getElementById('search-input').value.toLowerCase();
    filteredImages = images.filter(img => {
        // 搜索过滤
        if (search && !img.filename.toLowerCase().includes(search)) return false;
        // 状态过滤
        if (currentFilter === 'unlabeled') return !img.annotated;
        if (currentFilter === 'auto') return img.auto_labeled && !img.verified;
        if (currentFilter === 'verified') return img.verified;
        // 标签筛选过滤
        if (selectedTagFilters.length > 0) {
            // 只筛选已标注的图片
            if (!img.annotated) return false;
            // 获取图片的所有标签（包括labels和custom_tags）
            const imgLabels = img.labels || {};
            const allTags = new Set();
            // 收集所有标签值
            for (const [category, value] of Object.entries(imgLabels)) {
                if (Array.isArray(value)) {
                    value.forEach(v => allTags.add(v));
                } else if (value) {
                    allTags.add(value);
                }
            }
            // 收集自定义标签
            if (img.custom_tags && Array.isArray(img.custom_tags)) {
                img.custom_tags.forEach(tag => allTags.add(tag));
            }
            // 收集角色名称（从labels['角色名称']中）
            if (imgLabels['角色名称'] && Array.isArray(imgLabels['角色名称'])) {
                imgLabels['角色名称'].forEach(role => allTags.add(role));
            }
            // 检查是否包含所有选中的筛选标签
            for (const filterTag of selectedTagFilters) {
                if (!allTags.has(filterTag)) return false;
            }
        }
        return true;
    });
}

function setFilter(filter, btn) {
    currentFilter = filter;
    document.querySelectorAll('.filter-btns button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyFilter();
    renderImageList();
    if (filteredImages.length > 0) {
        selectImage(0);
    }
}

function filterImages() {
    applyFilter();
    renderImageList();
}

// ============================================================
// 标签筛选功能
// ============================================================
function onTagFilterInput() {
    const input = document.getElementById('tag-filter-input');
    const dropdown = document.getElementById('tag-filter-dropdown');
    const query = input.value.trim().toLowerCase();

    if (!query) {
        dropdown.style.display = 'none';
        return;
    }

    // 从labelConfig中收集所有可用标签
    const results = [];
    for (const [category, info] of Object.entries(labelConfig)) {
        for (const label of info.labels) {
            if (label.toLowerCase().includes(query) && !selectedTagFilters.includes(label)) {
                results.push({ label, category, type: 'label' });
            }
        }
    }

    // 从roleNamesData中收集角色名称
    const roleResults = [];
    for (const [source, names] of Object.entries(roleNamesData)) {
        for (const name of names) {
            if (name.toLowerCase().includes(query) && !selectedTagFilters.includes(name)) {
                roleResults.push({ label: name, category: source, type: 'role' });
            }
        }
    }

    // 合并结果，限制总数
    const allResults = [...results, ...roleResults].slice(0, 50);

    if (allResults.length === 0) {
        dropdown.style.display = 'none';
        return;
    }

    dropdown.innerHTML = '';
    let lastCategory = '';
    allResults.forEach(({ label, category, type }) => {
        const displayCategory = type === 'role' ? `角色名称 (${category})` : category;
        if (displayCategory !== lastCategory) {
            const group = document.createElement('div');
            group.className = 'tag-filter-group';
            group.textContent = displayCategory;
            dropdown.appendChild(group);
            lastCategory = displayCategory;
        }
        const item = document.createElement('div');
        item.className = 'tag-filter-item';
        item.innerHTML = `<span>${escapeHtml(label)}</span>${type === 'role' ? '<span class="count">角色</span>' : ''}`;
        item.onclick = () => {
            addTagFilter(label);
            input.value = '';
            dropdown.style.display = 'none';
        };
        dropdown.appendChild(item);
    });
    dropdown.style.display = 'block';
}

function addTagFilter(label) {
    if (!selectedTagFilters.includes(label)) {
        // 第一次添加筛选时，记住当前选中的图片
        if (selectedTagFilters.length === 0 && filteredImages.length > 0) {
            lastSelectedImageBeforeFilter = filteredImages[currentIndex]?.filename;
        }
        selectedTagFilters.push(label);
        renderSelectedTagFilters();
        applyFilter();
        renderImageList();
        if (filteredImages.length > 0) {
            selectImage(0);
        }
    }
}

function removeTagFilter(label) {
    selectedTagFilters = selectedTagFilters.filter(t => t !== label);
    renderSelectedTagFilters();
    applyFilter();
    renderImageList();
    if (filteredImages.length > 0) {
        // 如果清空了所有筛选条件，尝试恢复之前选中的图片
        if (selectedTagFilters.length === 0 && lastSelectedImageBeforeFilter) {
            const prevIndex = filteredImages.findIndex(img => img.filename === lastSelectedImageBeforeFilter);
            if (prevIndex >= 0) {
                selectImage(prevIndex);
                lastSelectedImageBeforeFilter = null; // 恢复后清空记忆
                return;
            }
            lastSelectedImageBeforeFilter = null; // 找不到也清空记忆
        }
        selectImage(0);
    }
}

function renderSelectedTagFilters() {
    const container = document.getElementById('selected-tag-filters');
    if (!container) return;
    container.innerHTML = '';
    selectedTagFilters.forEach(label => {
        const tag = document.createElement('span');
        tag.className = 'selected-tag-filter';
        tag.innerHTML = `${escapeHtml(label)} <span class="remove" onclick="removeTagFilter('${label.replace(/'/g, "\\'")}')">&times;</span>`;
        container.appendChild(tag);
    });
}

// 点击外部关闭下拉框
document.addEventListener('click', (e) => {
    const tagFilterBox = document.querySelector('.tag-filter-box');
    const dropdown = document.getElementById('tag-filter-dropdown');
    if (tagFilterBox && dropdown && !tagFilterBox.contains(e.target)) {
        dropdown.style.display = 'none';
    }
});

// ============================================================
// 渲染
// ============================================================
function renderImageList() {
    const list = document.getElementById('image-list');
    list.innerHTML = '';
    filteredImages.forEach((img, i) => {
        const div = document.createElement('div');
        div.className = 'image-item' + (i === currentIndex ? ' active' : '');
        div.onclick = () => selectImage(i);

        let dotClass = 'unlabeled';
        if (img.annotated && img.verified) dotClass = 'verified';
        else if (img.auto_labeled) dotClass = 'auto';

        div.innerHTML = `
            <img class="thumb" src="/api/image/${encodeURIComponent(img.filename)}" loading="lazy">
            <div class="info"><div class="name">${img.filename}</div></div>
            <div class="status-dot ${dotClass}"></div>
        `;
        list.appendChild(div);
    });
}

async function selectImage(index) {
    if (index < 0 || index >= filteredImages.length) return;
    currentIndex = index;
    const img = filteredImages[index];

    document.getElementById('main-image').src = '/api/image/' + encodeURIComponent(img.filename);
    document.getElementById('image-counter').textContent = `${index + 1} / ${filteredImages.length}`;

    currentLabels = img.labels ? JSON.parse(JSON.stringify(img.labels)) : {};
    currentCustomTags = [];
    // 切换图片时清空全局审核历史，让loadAnnotation加载新图片的审核历史
    window._currentReviewHistory = [];
    await loadAnnotation(img.filename);

    document.querySelectorAll('.image-item').forEach((el, i) => {
        el.classList.toggle('active', i === index);
    });

    const list = document.getElementById('image-list');
    const active = list.children[index];
    if (active) active.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

async function loadAnnotation(filename) {
    const res = await fetch('/api/annotation/' + encodeURIComponent(filename));
    const ann = await res.json();
    if (ann) {
        currentLabels = ann.labels || {};
        currentCustomTags = ann.custom_tags || [];
        document.getElementById('btn-verify').style.display = ann.verified ? 'none' : '';
        document.getElementById('label-title').textContent = ann.auto_labeled ? '(自动标注) 请校正' : '标签标注';
    } else {
        currentLabels = {};
        currentCustomTags = [];
        document.getElementById('btn-verify').style.display = 'none';
        document.getElementById('label-title').textContent = '标签标注';
    }
    renderLabelPanel();
    loadDescriptionIntoUI(ann && ann.description ? ann.description : '');
    loadReviewIntoUI(ann && ann.review ? ann.review : '', ann && ann.review_history ? ann.review_history : []);
    loadPoseIntoUI(ann && ann.pose ? ann.pose : null);
}

// ============================================================
// 角色名称搜索
// ============================================================
function renderRoleSearch(container) {
    selectedRoles = currentLabels['角色名称'] ? [...currentLabels['角色名称']] : [];
    if (!Array.isArray(selectedRoles)) selectedRoles = [];

    const box = document.createElement('div');
    box.className = 'role-search-box';
    box.innerHTML = `
        <h4>角色名称</h4>
        <input type="text" class="role-search-input" id="role-search-input" placeholder="搜索角色名..." autocomplete="off">
        <div class="selected-roles" id="selected-roles"></div>
    `;
    container.appendChild(box);

    // 创建全局下拉框（如果还没有）
    let dropdown = document.getElementById('global-role-dropdown');
    if (!dropdown) {
        dropdown = document.createElement('div');
        dropdown.className = 'role-dropdown';
        dropdown.id = 'global-role-dropdown';
        document.body.appendChild(dropdown);
    }

    // 渲染已选角色
    renderSelectedRoles();

    // 搜索输入事件
    const input = document.getElementById('role-search-input');

    function updateDropdownPosition() {
        const rect = input.getBoundingClientRect();
        dropdown.style.left = rect.left + 'px';
        dropdown.style.top = rect.bottom + 'px';
        dropdown.style.width = rect.width + 'px';
    }

    input.addEventListener('input', () => {
        const query = input.value.trim();
        if (!query) {
            dropdown.style.display = 'none';
            return;
        }
        const results = searchRoles(query);
        if (results.length === 0) {
            dropdown.style.display = 'none';
            return;
        }
        dropdown.innerHTML = '';
        let lastSource = '';
        results.forEach(({name, source}) => {
            if (source !== lastSource) {
                const group = document.createElement('div');
                group.className = 'role-dropdown-group';
                group.textContent = source;
                dropdown.appendChild(group);
                lastSource = source;
            }
            const item = document.createElement('div');
            item.className = 'role-dropdown-item';
            item.innerHTML = `<span class="role-name">${escapeHtml(name)}</span><span class="source-tag">${escapeHtml(source)}</span>`;
            item.onclick = () => {
                selectRole(name);
                input.value = '';
                dropdown.style.display = 'none';
            };
            dropdown.appendChild(item);
        });
        updateDropdownPosition();
        dropdown.style.display = 'block';
    });

    input.addEventListener('focus', () => {
        if (input.value.trim()) {
            updateDropdownPosition();
            input.dispatchEvent(new Event('input'));
        }
    });

    // 点击外部关闭下拉
    document.addEventListener('click', (e) => {
        if (!box.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.style.display = 'none';
        }
    });

    // 滚动时更新下拉框位置
    const scrollContainer = document.getElementById('section-labels');
    if (scrollContainer) {
        scrollContainer.addEventListener('scroll', () => {
            if (dropdown.style.display === 'block') {
                updateDropdownPosition();
            }
        });
    }
}

function searchRoles(query) {
    const q = query.toLowerCase();
    const results = [];
    for (const [source, names] of Object.entries(roleNamesData)) {
        for (const name of names) {
            if (name.toLowerCase().includes(q) || source.toLowerCase().includes(q)) {
                results.push({name, source});
            }
        }
    }
    return results.slice(0, 50);
}

function selectRole(name) {
    if (!selectedRoles.includes(name)) {
        selectedRoles.push(name);
    }
    currentLabels['角色名称'] = [...selectedRoles];
    renderSelectedRoles();
}

function removeRole(name) {
    selectedRoles = selectedRoles.filter(r => r !== name);
    currentLabels['角色名称'] = [...selectedRoles];
    renderSelectedRoles();
}

function renderSelectedRoles() {
    const container = document.getElementById('selected-roles');
    if (!container) return;
    container.innerHTML = '';
    selectedRoles.forEach(name => {
        const tag = document.createElement('span');
        tag.className = 'selected-role-tag';
        tag.innerHTML = `${escapeHtml(name)} <span class="remove" onclick="removeRole('${name.replace(/'/g, "\\'")}')">&times;</span>`;
        container.appendChild(tag);
    });
}

function renderLabelPanel() {
    const body = document.getElementById('label-body');
    body.innerHTML = '';

    // 角色名称搜索区域（放在最顶部）
    if (Object.keys(roleNamesData).length > 0) {
        renderRoleSearch(body);
    }

    for (const [category, info] of Object.entries(labelConfig)) {
        const catDiv = document.createElement('div');
        catDiv.className = 'category';
        const mode = info.multi ? '多选' : '单选';
        catDiv.innerHTML = `<div class="category-title">${category} <span class="mode">${mode}</span></div>`;

        const grid = document.createElement('div');
        grid.className = 'tag-grid';

        const selected = currentLabels[category];
        const isAuto = filteredImages[currentIndex]?.auto_labeled;

        info.labels.forEach(label => {
            const tag = document.createElement('span');
            tag.className = 'tag';
            if (isAuto) tag.classList.add('auto-tag');

            // 检查是否选中
            if (info.multi) {
                const arr = selected || [];
                if (arr.includes(label)) tag.classList.add('selected');
            } else {
                if (selected === label) tag.classList.add('selected');
            }

            tag.textContent = label;
            tag.onclick = () => toggleTag(category, label, info.multi);
            grid.appendChild(tag);
        });

        catDiv.appendChild(grid);
        body.appendChild(catDiv);
    }

    // 自定义标签区域
    const customDiv = document.createElement('div');
    customDiv.className = 'custom-tags';
    customDiv.innerHTML = `
        <h4>自定义标签</h4>
        <div class="custom-tag-input">
            <input type="text" id="custom-tag-input" placeholder="输入自定义标签，回车添加" onkeydown="if(event.key==='Enter')addCustomTag()">
            <button class="btn btn-outline" onclick="addCustomTag()">添加</button>
        </div>
        <div class="custom-tag-list" id="custom-tag-list"></div>
    `;
    body.appendChild(customDiv);

    document.getElementById('desc-box').style.display = 'block';
    document.getElementById('pose-box').style.display = 'block';
    document.getElementById('review-box').style.display = 'block';
    renderCustomTags();
}

function renderCustomTags() {
    const list = document.getElementById('custom-tag-list');
    if (!list) return;
    list.innerHTML = '';
    currentCustomTags.forEach(tag => {
        const el = document.createElement('span');
        el.className = 'custom-tag';
        el.innerHTML = `${tag} <span class="remove" onclick="removeCustomTag('${tag}')">&times;</span>`;
        list.appendChild(el);
    });
}

async function generateDescription() {
    if (filteredImages.length === 0) return;
    const img = filteredImages[currentIndex];
    showToast('正在生成描述...', 'info');
    try {
        const res = await fetch('/api/generate-description/' + encodeURIComponent(img.filename), { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('desc-placeholder').style.display = 'none';
            document.getElementById('desc-text').style.display = 'inline';
            document.getElementById('desc-text').textContent = data.description;
            document.getElementById('desc-textarea').value = data.description;
            document.getElementById('desc-actions').style.display = 'block';
            showToast('描述生成成功', 'success');
        } else {
            showToast(data.error || '生成描述失败', 'error');
        }
    } catch (e) {
        showToast('生成描述失败: ' + e.message, 'error');
    }
}

async function generateSemiFreeDescription() {
    if (filteredImages.length === 0) return;
    const img = filteredImages[currentIndex];
    showToast('正在调用模型生成描述...', 'info');
    try {
        const res = await fetch('/api/generate-semi-free-description/' + encodeURIComponent(img.filename), { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            document.getElementById('desc-placeholder').style.display = 'none';
            document.getElementById('desc-text').style.display = 'inline';
            document.getElementById('desc-text').textContent = data.description;
            document.getElementById('desc-textarea').value = data.description;
            document.getElementById('desc-actions').style.display = 'block';
            showToast('描述生成成功', 'success');
        } else {
            showToast(data.error || '生成描述失败', 'error');
        }
    } catch (e) {
        showToast('生成描述失败: ' + e.message, 'error');
    }
}

async function saveDescription() {
    if (filteredImages.length === 0) return;
    const img = filteredImages[currentIndex];
    const descText = document.getElementById('desc-textarea').value.trim();
    if (!descText) {
        showToast('描述内容不能为空', 'error');
        return;
    }
    try {
        const res = await fetch('/api/annotation/' + encodeURIComponent(img.filename), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                labels: currentLabels,
                custom_tags: currentCustomTags,
                description: descText,
                review: document.getElementById('review-textarea')?.value || '',
                auto_labeled: img.auto_labeled || false,
                verified: img.verified || false
            })
        });
        if (res.ok) {
            document.getElementById('desc-text').textContent = descText;
            showToast('描述已保存', 'success');
        } else {
            showToast('保存描述失败', 'error');
        }
    } catch (e) {
        showToast('保存描述失败: ' + e.message, 'error');
    }
}

async function generateReview() {
    if (filteredImages.length === 0) {
        showToast('请先选择一张图片', 'error');
        return;
    }
    const img = filteredImages[currentIndex];
    if (!img) {
        showToast('图片不存在', 'error');
        return;
    }

    const chat = document.getElementById('review-chat');
    if (!chat) {
        showToast('审核区域未初始化，请重试', 'error');
        return;
    }

    showToast('正在生成审核结果...', 'info');
    try {
        const res = await fetch('/api/generate-review/' + encodeURIComponent(img.filename), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });
        const data = await res.json();
        if (res.ok) {
            const inputRow = document.getElementById('review-input-row');
            if (inputRow) inputRow.style.display = 'flex';
            renderReviewHistory(data.history || []);
            showToast('审核生成成功', 'success');
        } else {
            showToast(data.error || '生成审核失败', 'error');
        }
    } catch (e) {
        showToast('生成审核失败: ' + e.message, 'error');
    }
}

async function sendReviewMessage() {
    if (filteredImages.length === 0) {
        showToast('请先选择一张图片', 'error');
        return;
    }
    const img = filteredImages[currentIndex];
    if (!img) {
        showToast('图片不存在', 'error');
        return;
    }
    const userInput = document.getElementById('review-user-input');
    if (!userInput) {
        showToast('输入框未初始化', 'error');
        return;
    }
    const message = userInput.value.trim();
    if (!message) return;

    userInput.value = '';
    const chat = document.getElementById('review-chat');
    if (!chat) {
        showToast('聊天区域未初始化', 'error');
        return;
    }

    const userMsg = document.createElement('div');
    userMsg.className = 'review-msg user';
    userMsg.innerHTML = `<div class="role">用户</div><div>${escapeHtml(message)}</div>`;
    chat.appendChild(userMsg);
    chat.scrollTop = chat.scrollHeight;

    showToast('正在等待回复...', 'info');
    try {
        const res = await fetch('/api/generate-review/' + encodeURIComponent(img.filename), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_input: message })
        });
        const data = await res.json();
        if (res.ok) {
            renderReviewHistory(data.history || []);
            chat.scrollTop = chat.scrollHeight;
            showToast('回复成功', 'success');
        } else {
            showToast(data.error || '回复失败', 'error');
        }
    } catch (e) {
        showToast('回复失败: ' + e.message, 'error');
    }
}

function renderReviewHistory(history) {
    window._currentReviewHistory = history || [];
    const chat = document.getElementById('review-chat');
    if (!chat) return;

    chat.innerHTML = '';
    if (!history || history.length === 0) {
        chat.innerHTML = '<span class="review-placeholder" id="review-placeholder">暂无审核结果，点击"生成审核"按钮根据标签自动生成</span>';
        return;
    }
    history.forEach(msg => {
        const div = document.createElement('div');
        div.className = 'review-msg ' + msg.role;
        div.innerHTML = `<div class="role">${msg.role === 'user' ? '用户' : '助手'}</div><div>${escapeHtml(msg.content)}</div>`;
        chat.appendChild(div);
    });
}

function loadReviewIntoUI(review, history) {
    const placeholder = document.getElementById('review-placeholder');
    const inputRow = document.getElementById('review-input-row');

    if (review || (history && history.length > 0)) {
        if (placeholder) placeholder.style.display = 'none';
        if (inputRow) inputRow.style.display = 'flex';
        renderReviewHistory(history || []);
    } else {
        if (placeholder) placeholder.style.display = 'inline';
        if (inputRow) inputRow.style.display = 'none';
        const chat = document.getElementById('review-chat');
        if (chat) chat.innerHTML = '<span class="review-placeholder" id="review-placeholder">暂无审核结果，点击"生成审核"按钮根据标签自动生成</span>';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function loadDescriptionIntoUI(description) {
    const placeholder = document.getElementById('desc-placeholder');
    const descText = document.getElementById('desc-text');
    const descTextarea = document.getElementById('desc-textarea');
    const descActions = document.getElementById('desc-actions');

    if (description) {
        if (placeholder) placeholder.style.display = 'none';
        if (descText) {
            descText.style.display = 'inline';
            descText.textContent = description;
        }
        if (descTextarea) descTextarea.value = description;
        if (descActions) descActions.style.display = 'block';
    } else {
        if (placeholder) placeholder.style.display = 'inline';
        if (descText) {
            descText.style.display = 'none';
            descText.textContent = '';
        }
        if (descTextarea) descTextarea.value = '';
        if (descActions) descActions.style.display = 'none';
    }
}

// ============================================================
// 标注操作
// ============================================================
function toggleTag(category, label, multi) {
    if (multi) {
        if (!currentLabels[category]) currentLabels[category] = [];
        const idx = currentLabels[category].indexOf(label);
        if (idx >= 0) {
            currentLabels[category].splice(idx, 1);
        } else {
            currentLabels[category].push(label);
        }
    } else {
        currentLabels[category] = currentLabels[category] === label ? '' : label;
    }
    const descText = document.getElementById('desc-textarea')?.value || '';
    // 保存当前的审核历史，在renderLabelPanel后恢复
    const currentReviewHistory = window._currentReviewHistory ? [...window._currentReviewHistory] : [];
    // 保存姿态图像状态
    const poseImgSrc = document.getElementById('pose-image')?.src || '';
    const poseActionsDisplay = document.getElementById('pose-actions')?.style.display || 'none';
    renderLabelPanel();
    loadDescriptionIntoUI(descText);
    // 恢复审核历史显示，不清空已保存的审核结果
    window._currentReviewHistory = currentReviewHistory;
    loadReviewIntoUI('', currentReviewHistory);
    // 恢复姿态图像
    if (poseImgSrc) {
        const poseImg = document.getElementById('pose-image');
        if (poseImg) {
            poseImg.src = poseImgSrc;
            poseImg.style.display = 'block';
        }
        const posePlaceholder = document.getElementById('pose-placeholder');
        if (posePlaceholder) posePlaceholder.style.display = 'none';
    }
    const poseActions = document.getElementById('pose-actions');
    if (poseActions) poseActions.style.display = poseActionsDisplay;
}

function addCustomTag() {
    const input = document.getElementById('custom-tag-input');
    const tag = input.value.trim();
    if (tag && !currentCustomTags.includes(tag)) {
        currentCustomTags.push(tag);
        input.value = '';
        renderCustomTags();
    }
}

function removeCustomTag(tag) {
    currentCustomTags = currentCustomTags.filter(t => t !== tag);
    renderCustomTags();
}

function clearCurrent() {
    currentLabels = {};
    currentCustomTags = [];
    renderLabelPanel();
}

// ============================================================
// 保存
// ============================================================
async function saveCurrent() {
    if (filteredImages.length === 0) return;
    const img = filteredImages[currentIndex];
    try {
        const res = await fetch('/api/annotation/' + encodeURIComponent(img.filename), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                labels: currentLabels,
                custom_tags: currentCustomTags,
                description: document.getElementById('desc-textarea')?.value || '',
                auto_labeled: img.auto_labeled || false,
                verified: false
            })
        });
        if (res.ok) {
            showToast('已保存', 'success');
            await loadImages();
            await loadStats();
            renderImageList();
            await selectImage(currentIndex);
        }
    } catch (e) {
        showToast('保存失败: ' + e.message, 'error');
    }
}

async function verifyCurrent() {
    if (filteredImages.length === 0) return;
    const img = filteredImages[currentIndex];
    await fetch('/api/verify/' + encodeURIComponent(img.filename), { method: 'POST' });
    showToast('已确认', 'success');
    await loadImages();
    await loadStats();
    renderImageList();
    navigate(1);
}

// ============================================================
// 自动标注
// ============================================================
async function autoLabelCurrent() {
    if (filteredImages.length === 0) return;
    const img = filteredImages[currentIndex];
    showToast('正在自动标注...', 'info');
    try {
        const res = await fetch('/api/auto-label/' + encodeURIComponent(img.filename), { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
            currentLabels = data.labels || {};
            renderLabelPanel();
            showToast('自动标注完成，请检查', 'success');
            await loadImages();
            await loadStats();
            renderImageList();
            await selectImage(currentIndex);
        } else {
            showToast(data.error || '自动标注失败', 'error');
        }
    } catch (e) {
        showToast('自动标注失败: ' + e.message, 'error');
    }
}

async function startBatchAutoLabel() {
    const batchSize = prompt('请输入批量标注数量 (建议20-50张):', '20');
    if (!batchSize) return;
    try {
        const res = await fetch('/api/auto-label-batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ batch_size: parseInt(batchSize), overwrite: false })
        });
        const data = await res.json();
        if (res.ok) {
            showToast(`开始批量标注 ${data.total} 张图片`, 'info');
            document.getElementById('batch-progress').classList.add('show');
            document.getElementById('batch-total').textContent = data.total;
            pollBatchProgress();
        } else {
            showToast(data.error || '批量标注失败', 'error');
        }
    } catch (e) {
        showToast('请求失败: ' + e.message, 'error');
    }
}

async function pollBatchProgress() {
    try {
        const res = await fetch('/api/auto-label-progress');
        const p = await res.json();
        document.getElementById('batch-done').textContent = p.done;
        const pct = p.total > 0 ? (p.done / p.total * 100) : 0;
        document.getElementById('batch-bar').style.width = pct + '%';
        if (p.running) {
            setTimeout(pollBatchProgress, 1000);
        } else {
            showToast('批量标注完成!', 'success');
            document.getElementById('batch-progress').classList.remove('show');
            await loadImages();
            await loadStats();
            renderImageList();
            await selectImage(currentIndex);
        }
    } catch {
        setTimeout(pollBatchProgress, 2000);
    }
}

// ============================================================
// 导航
// ============================================================
async function navigate(delta) {
    const newIdx = currentIndex + delta;
    if (newIdx >= 0 && newIdx < filteredImages.length) {
        await selectImage(newIdx);
    }
}

// ============================================================
// 导出
// ============================================================
function exportAnnotations(format) {
    window.open('/api/export?format=' + format, '_blank');
}

// ============================================================
// 快捷键
// ============================================================
function toggleShortcuts() {
    document.getElementById('shortcuts').classList.toggle('show');
}

// ============================================================
// 打开文件夹功能
// ============================================================
async function openImageFolder() {
    if (filteredImages.length === 0) {
        showToast('请先选择一张图片', 'error');
        return;
    }
    const img = filteredImages[currentIndex];
    try {
        const res = await fetch('/api/open-folder/image/' + encodeURIComponent(img.filename), { method: 'POST' });
        if (res.ok) {
            showToast('已打开图片所在文件夹', 'success');
        } else {
            const data = await res.json();
            showToast(data.error || '打开文件夹失败', 'error');
        }
    } catch (e) {
        showToast('打开文件夹失败: ' + e.message, 'error');
    }
}

async function openAnnotationsFolder() {
    try {
        const res = await fetch('/api/open-folder/annotations', { method: 'POST' });
        if (res.ok) {
            showToast('已打开标注文件所在文件夹', 'success');
        } else {
            const data = await res.json();
            showToast(data.error || '打开文件夹失败', 'error');
        }
    } catch (e) {
        showToast('打开文件夹失败: ' + e.message, 'error');
    }
}

async function deleteCurrentImage() {
    if (filteredImages.length === 0) {
        showToast('请先选择一张图片', 'error');
        return;
    }
    const img = filteredImages[currentIndex];
    const hasAnnotation = img.annotated || img.auto_labeled;
    const confirmMsg = hasAnnotation
        ? `确定要删除图片 "${img.filename}" 吗？\n\n该图片已有标注内容，删除后将同时删除标注数据。\n此操作不可恢复！`
        : `确定要删除图片 "${img.filename}" 吗？\n\n此操作不可恢复！`;

    if (!confirm(confirmMsg)) {
        return;
    }

    try {
        const res = await fetch('/api/image/' + encodeURIComponent(img.filename), { method: 'DELETE' });
        if (res.ok) {
            showToast('图片已删除', 'success');
            // 重新加载图片列表
            await loadImages();
            await loadStats();
            renderImageList();
            // 如果还有图片，选中合适的位置
            if (filteredImages.length > 0) {
                const newIndex = Math.min(currentIndex, filteredImages.length - 1);
                selectImage(Math.max(0, newIndex));
            } else {
                // 没有图片了，清空显示
                document.getElementById('main-image').src = '';
                document.getElementById('image-counter').textContent = '0 / 0';
                document.getElementById('label-body').innerHTML = '<p style="color:var(--text-dim); text-align:center; margin-top:40px;">没有图片</p>';
            }
        } else {
            const data = await res.json();
            showToast(data.error || '删除失败', 'error');
        }
    } catch (e) {
        showToast('删除失败: ' + e.message, 'error');
    }
}

document.addEventListener('keydown', (e) => {
    // Ctrl+S 保存
    if (e.ctrlKey && e.key === 's') { e.preventDefault(); saveCurrent(); }
    // Ctrl+Enter 保存并下一张
    if (e.ctrlKey && e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveCurrent().then(() => navigate(1)); }
    // Ctrl+Shift+Enter 确认并下一张
    if (e.ctrlKey && e.shiftKey && e.key === 'Enter') { e.preventDefault(); verifyCurrent(); }
    // Ctrl+1 自动标注
    if (e.ctrlKey && e.key === '1') { e.preventDefault(); autoLabelCurrent(); }
    // 左右箭头
    if (e.key === 'ArrowLeft' && !e.ctrlKey && document.activeElement.tagName !== 'INPUT') navigate(-1);
    if (e.key === 'ArrowRight' && !e.ctrlKey && document.activeElement.tagName !== 'INPUT') navigate(1);
    // Q 快捷键提示
    if (e.key === 'q' && document.activeElement.tagName !== 'INPUT') toggleShortcuts();
});

// ============================================================
// 姿态估计
// ============================================================
function loadPoseIntoUI(poseData) {
    const placeholder = document.getElementById('pose-placeholder');
    const poseImg = document.getElementById('pose-image');
    const poseActions = document.getElementById('pose-actions');

    if (poseData && poseData.pose_image_path) {
        if (placeholder) placeholder.style.display = 'none';
        if (poseImg) {
            poseImg.src = '/api/pose-image/' + encodeURIComponent(poseData.pose_image_path);
            poseImg.style.display = 'block';
        }
        if (poseActions) poseActions.style.display = 'flex';
    } else {
        if (placeholder) {
            placeholder.style.display = 'inline';
            placeholder.textContent = '暂无姿态数据，点击上方按钮识别';
        }
        if (poseImg) {
            poseImg.style.display = 'none';
            poseImg.src = '';
        }
        if (poseActions) poseActions.style.display = 'none';
    }
}

async function estimatePose() {
    if (filteredImages.length === 0) {
        showToast('请先选择一张图片', 'error');
        return;
    }
    const img = filteredImages[currentIndex];

    const container = document.getElementById('pose-image-container');
    const placeholder = document.getElementById('pose-placeholder');
    const poseImg = document.getElementById('pose-image');
    const poseActions = document.getElementById('pose-actions');

    if (placeholder) placeholder.style.display = 'none';
    if (poseImg) poseImg.style.display = 'none';

    // 显示加载动画
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'pose-loading';
    loadingDiv.id = 'pose-loading';
    loadingDiv.innerHTML = '<div class="spinner"></div> 正在识别姿态...';
    container.appendChild(loadingDiv);

    showToast('正在识别姿态，请稍候...', 'info');

    try {
        const res = await fetch('/api/pose-estimate/' + encodeURIComponent(img.filename), { method: 'POST' });
        const data = await res.json();

        const loading = document.getElementById('pose-loading');
        if (loading) loading.remove();

        if (res.ok) {
            if (poseImg) {
                poseImg.src = 'data:image/png;base64,' + data.pose_image_b64;
                poseImg.style.display = 'block';
            }
            if (poseActions) poseActions.style.display = 'flex';
            showToast('姿态识别完成，请确认保存', 'success');
        } else {
            if (placeholder) {
                placeholder.style.display = 'inline';
                placeholder.textContent = data.error || '姿态识别失败';
            }
            showToast(data.error || '姿态识别失败', 'error');
        }
    } catch (e) {
        const loading = document.getElementById('pose-loading');
        if (loading) loading.remove();
        if (placeholder) {
            placeholder.style.display = 'inline';
            placeholder.textContent = '识别失败: ' + e.message;
        }
        showToast('姿态识别失败: ' + e.message, 'error');
    }
}

async function confirmPoseSave() {
    if (filteredImages.length === 0) return;
    if (!confirm('确认保存姿态数据？')) return;

    const img = filteredImages[currentIndex];
    await loadImages();
    await selectImage(currentIndex);
    showToast('姿态数据已保存', 'success');
}

async function clearPose() {
    if (filteredImages.length === 0) return;
    const img = filteredImages[currentIndex];
    try {
        const res = await fetch('/api/pose/' + encodeURIComponent(img.filename), { method: 'DELETE' });
        if (res.ok) {
            loadPoseIntoUI(null);
            showToast('姿态数据已清除', 'success');
        }
    } catch (e) {
        showToast('清除失败: ' + e.message, 'error');
    }
}

// ============================================================
// Toast
// ============================================================
function showToast(msg, type = 'info') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = 'toast show ' + type;
    setTimeout(() => t.className = 'toast', 2500);
}

// ============================================================
// 悬浮功能切换
// ============================================================
const sectionVisibility = {
    labels: true,
    pose: false,
    desc: false,
    review: false
};

function toggleSection(section) {
    sectionVisibility[section] = !sectionVisibility[section];
    updateSectionVisibility();
    updateToolbarButtons();
}

function updateSectionVisibility() {
    const sectionNames = ['labels', 'pose', 'desc', 'review'];
    const container = document.getElementById('resizable-container');
    if (!container) return;

    const allSections = {
        labels: document.getElementById('section-labels'),
        pose: document.getElementById('section-pose'),
        desc: document.getElementById('section-desc'),
        review: document.getElementById('section-review')
    };
    const handleElements = [
        document.getElementById('handle-0'),
        document.getElementById('handle-1'),
        document.getElementById('handle-2')
    ];

    // 获取可见section的名称和索引
    const visibleInfo = [];
    sectionNames.forEach((name, idx) => {
        if (sectionVisibility[name]) visibleInfo.push({ name, idx });
    });
    const visibleSections = visibleInfo.map(v => v.name);

    // 设置section显示
    sectionNames.forEach(name => {
        const sec = allSections[name];
        if (sec) sec.style.display = sectionVisibility[name] ? '' : 'none';
    });

    // 无可见section
    if (visibleSections.length === 0) {
        handleElements.forEach(h => { if (h) h.style.display = 'none'; });
        return;
    }

    // 清空所有handle的data和display
    handleElements.forEach(h => {
        if (h) {
            h.style.display = 'none';
            h.dataset.topSection = '';
            h.dataset.bottomSection = '';
        }
    });

    // 为每对相邻可见section选择正确的handle
    // DOM结构: section-0, handle-0, section-1, handle-1, section-2, handle-2, section-3
    // handle-i 紧跟在 section-i 后面
    // 对于位置a和b(b>a)的两个相邻可见section，显示handle-a
    // 因为a和b之间的所有section都隐藏了(display:none不占空间)，
    // handle-a会自然出现在section-a和section-b之间
    for (let i = 0; i < visibleInfo.length - 1; i++) {
        const topIdx = visibleInfo[i].idx;
        const handle = handleElements[topIdx];
        if (handle) {
            handle.style.display = '';
            handle.dataset.topSection = visibleInfo[i].name;
            handle.dataset.bottomSection = visibleInfo[i + 1].name;
        }
    }

    // 强制reflow后设置高度
    void container.offsetHeight;
    const visibleHandleCount = handleElements.filter(h => h && h.style.display !== 'none').length;
    const totalH = container.clientHeight - visibleHandleCount * 6;
    const sectionHeight = Math.max(80, Math.floor(totalH / visibleSections.length));

    visibleSections.forEach(name => {
        const sec = allSections[name];
        if (sec) {
            sec.style.height = sectionHeight + 'px';
            sec.style.overflowY = 'auto';
        }
    });
}

function updateToolbarButtons() {
    const buttons = {
        labels: document.getElementById('btn-toggle-labels'),
        pose: document.getElementById('btn-toggle-pose'),
        desc: document.getElementById('btn-toggle-desc'),
        review: document.getElementById('btn-toggle-review')
    };

    for (const [name, btn] of Object.entries(buttons)) {
        if (btn) {
            btn.classList.toggle('active', sectionVisibility[name]);
        }
    }
}

// ============================================================
// 可拖拽调整布局
// ============================================================
function initResizableLayout() {
    const container = document.getElementById('resizable-container');
    if (!container) return;

    const MIN_HEIGHT = 60;
    const sectionNames = ['labels', 'pose', 'desc', 'review'];

    // 获取所有section和handle元素
    const allSections = sectionNames.map(name => document.getElementById('section-' + name));
    const allHandles = [0, 1, 2].map(i => document.getElementById('handle-' + i));

    // 获取当前可见的sections
    function getVisibleSections() {
        return sectionNames
            .map((name, index) => ({ name, element: allSections[index], index }))
            .filter(item => sectionVisibility[item.name]);
    }

    // 根据handle的data属性动态获取它控制的一对section
    function getAdjacentSections(handleIndex) {
        const handle = allHandles[handleIndex];
        if (!handle || handle.style.display === 'none') return null;
        const topName = handle.dataset.topSection;
        const bottomName = handle.dataset.bottomSection;
        if (!topName || !bottomName) return null;
        const topSec = document.getElementById('section-' + topName);
        const bottomSec = document.getElementById('section-' + bottomName);
        if (!topSec || !bottomSec) return null;
        return { topSec, bottomSec };
    }

    // 为handle绑定拖拽事件
    function bindHandleEvents(handleIndex) {
        const handle = allHandles[handleIndex];
        if (!handle) return;

        let startY = 0;
        let startTopH = 0;
        let startBottomH = 0;

        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();

            const adj = getAdjacentSections(handleIndex);
            if (!adj) return;

            const topSec = adj.topSec;
            const bottomSec = adj.bottomSec;

            startY = e.clientY;
            startTopH = topSec.offsetHeight;
            startBottomH = bottomSec.offsetHeight;
            handle.classList.add('active');
            document.body.style.cursor = 'row-resize';
            document.body.style.userSelect = 'none';

            const onMouseMove = (e) => {
                const dy = e.clientY - startY;
                let newTopH = startTopH + dy;
                let newBottomH = startBottomH - dy;

                if (newTopH < MIN_HEIGHT) {
                    newTopH = MIN_HEIGHT;
                    newBottomH = startTopH + startBottomH - MIN_HEIGHT;
                }
                if (newBottomH < MIN_HEIGHT) {
                    newBottomH = MIN_HEIGHT;
                    newTopH = startTopH + startBottomH - MIN_HEIGHT;
                }

                topSec.style.height = newTopH + 'px';
                bottomSec.style.height = newBottomH + 'px';
            };

            const onMouseUp = () => {
                handle.classList.remove('active');
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
            };

            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });

        // 双击重置为平均高度
        handle.addEventListener('dblclick', () => {
            const visibleSections = getVisibleSections();
            if (visibleSections.length > 0) {
                const handleCount = Math.max(0, visibleSections.length - 1);
                const totalH = container.clientHeight - handleCount * 6;
                const sectionHeight = Math.max(MIN_HEIGHT, Math.floor(totalH / visibleSections.length));
                visibleSections.forEach(item => {
                    item.element.style.height = sectionHeight + 'px';
                });
            }
        });
    }

    // 初始化所有handle的拖拽事件（只绑定一次）
    allHandles.forEach((_, index) => bindHandleEvents(index));

    // 窗口大小变化时重新计算
    window.addEventListener('resize', () => {
        const visibleSections = getVisibleSections();
        if (visibleSections.length > 0) {
            const handleCount = Math.max(0, visibleSections.length - 1);
            const totalH = container.clientHeight - handleCount * 6;
            const sectionHeight = Math.max(MIN_HEIGHT, Math.floor(totalH / visibleSections.length));
            visibleSections.forEach(item => {
                item.element.style.height = sectionHeight + 'px';
            });
        }
    });
}

// ============================================================
// 启动
// ============================================================
init();
initResizableLayout();
updateSectionVisibility(); // 初始化功能面板显示状态
