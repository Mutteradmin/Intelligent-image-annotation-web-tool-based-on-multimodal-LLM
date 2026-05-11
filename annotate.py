"""
图像多标签标注工具 - 自动标注 + 人工校正

支持三种自动标注模式:
  1. 本地模型: python annotate.py --local-model F:/qwen3_5
  2. OpenAI API: python annotate.py --api-key YOUR_KEY --api-type openai
  3. Claude API: python annotate.py --api-key YOUR_KEY --api-type anthropic
  4. 仅手动:    python annotate.py
"""

import os
import sys
import json
import base64
import argparse
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request, render_template, send_from_directory
from PIL import Image

# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(__file__).parent
IMAGE_DIR = BASE_DIR
ANNOTATIONS_FILE = BASE_DIR / "annotations.json"
LABEL_CONFIG_FILE = BASE_DIR / "label_config.json"
LAST_POSITION_FILE = BASE_DIR / "last_position.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
POSE_SKELETONS_DIR = BASE_DIR / "pose_skeletons"

# ============================================================
# 标签配置 (预设分类)
# ============================================================
DEFAULT_LABEL_CONFIG = {
    "性别": {
        "labels": ["女性", "男性"],
        "multi": False
    },
    "发色": {
        "labels": ["白色", "黑色", "金色", "橙色", "粉色", "蓝色", "灰色", "棕色", "红色", "绿色", "紫色", "银色"],
        "multi": True
    },
    "发型": {
        "labels": ["长发", "短发", "马尾", "双马尾", "波波头", "卷发", "直发", "波浪", "丸子头", "散发"],
        "multi": True
    },
    "瞳色": {
        "labels": ["蓝色", "绿色", "红色", "棕色", "紫色", "黄色", "灰色", "黑色", "异色瞳"],
        "multi": True
    },
    "角色特征": {
        "labels": ["猫耳", "兽耳", "角", "翅膀", "尾巴", "眼镜", "帽子", "蝴蝶结", "发带", "光环", "精灵耳", "呆毛"],
        "multi": True
    },
    "服装": {
        "labels": ["女仆装", "校服", "泳装", "比基尼", "哥特", "铠甲", "休闲", "运动装", "裙子", "和服", "制服", "连衣裙", "圣诞装", "旗袍"],
        "multi": True
    },
    "姿势": {
        "labels": ["站立", "坐姿", "动态", "躺姿", "行走", "跑步", "蹲姿"],
        "multi": True
    },
    "背景": {
        "labels": ["白色背景", "深色背景", "室外", "室内", "简单背景", "复杂背景", "透明背景"],
        "multi": True
    },
    "画面风格": {
        "labels": ["动漫", "Q版", "写实", "水彩", "素描"],
        "multi": True
    },
    "人物数量": {
        "labels": ["单人", "双人", "多人"],
        "multi": False
    }
}


def load_label_config():
    """加载标签配置"""
    if LABEL_CONFIG_FILE.exists():
        with open(LABEL_CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    config = DEFAULT_LABEL_CONFIG
    save_label_config(config)
    return config


def save_label_config(config):
    with open(LABEL_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_annotations():
    """加载标注数据"""
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_annotations(data):
    """保存标注数据"""
    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_last_position():
    """加载上次查看位置"""
    if LAST_POSITION_FILE.exists():
        with open(LAST_POSITION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_last_position(filename):
    """保存当前查看位置"""
    data = {
        "filename": filename,
        "timestamp": datetime.now().isoformat()
    }
    with open(LAST_POSITION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_image_list():
    """获取所有图片文件名"""
    files = []
    for f in sorted(os.listdir(IMAGE_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            files.append(f)
    return files


def image_to_base64(filepath, max_size=512):
    """将图片转为 base64 (用于 API 调用)"""
    img = Image.open(filepath)
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ============================================================
# 自动标注 - 支持多种后端
# ============================================================

def auto_label_with_openai(filepath, api_key, model="gpt-4o", base_url=None):
    """使用 OpenAI 兼容 API 自动标注"""
    import urllib.request
    import urllib.error

    filename = os.path.basename(filepath)
    print(f"[OpenAI] 开始标注: {filename}")
    b64 = image_to_base64(filepath)
    config = load_label_config()

    categories_desc = [] # 没有明显特征的分类可以返回空数组[]
    for cat, info in config.items():
        mode = "单选" if not info["multi"] else "多选"
        labels = ", ".join(info["labels"])
        categories_desc.append(f"- {cat}({mode}): [{labels}]")

    prompt = f"""请分析这张图片，为以下每个分类选择最合适的标签。

分类列表:
{chr(10).join(categories_desc)} 

请严格按照以下示例的 JSON 格式输出，可能还有其他的标签，不要包含其他内容:
{{"性别": "标签", "发色": ["标签1"], "发型": ["标签1"], "瞳色": ["标签1"], "角色特征": ["标签1", "标签2"], "服装": ["标签1"], "姿势": ["标签1"], "背景": ["标签1"], "画面风格": ["标签1"], "人物数量": "标签"}}

注意:
- 单选分类用字符串, 多选分类用数组
- 只能从上面列出的标签中选择
- 每个标签必须选择一个类别符合图中情况的，不许跳过返回空
- 必须是纯JSON, 不要markdown代码块"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.2
    }

    url = (base_url or "https://api.openai.com") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    print(f"[OpenAI] 正在请求 API: {model}")
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())

    text = result["choices"][0]["message"]["content"].strip()
    print(f"[OpenAI] 收到响应，长度: {len(text)} 字符")
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    labels = json.loads(text)
    print(f"[OpenAI] 标注完成: {filename}")#-> {list(labels.keys())}
    return labels


def auto_label_with_anthropic(filepath, api_key, model="claude-sonnet-4-20250514"):
    """使用 Anthropic Claude API 自动标注"""
    import urllib.request

    filename = os.path.basename(filepath)
    print(f"[Anthropic] 开始标注: {filename}")
    b64 = image_to_base64(filepath)
    config = load_label_config()

    categories_desc = []
    for cat, info in config.items():
        mode = "单选" if not info["multi"] else "多选"
        labels = ", ".join(info["labels"])
        categories_desc.append(f"- {cat}({mode}): [{labels}]")

    prompt = f"""请分析这张图片，为以下每个分类选择最合适的标签。

分类列表:
{chr(10).join(categories_desc)}

请严格按照以下示例的 JSON 格式输出，可能还有其他的标签，不要包含其他内容:
{{"性别": "标签", "发色": ["标签1"], "发型": ["标签1"], "瞳色": ["标签1"], "角色特征": ["标签1", "标签2"], "服装": ["标签1"], "姿势": ["标签1"], "背景": ["标签1"], "画面风格": ["标签1"], "人物数量": "标签"}}

注意:
- 单选分类用字符串, 多选分类用数组
- 只能从上面列出的标签中选择
- 每个标签必须选择一个类别符合图中情况的，不许跳过返回空
- 必须是纯JSON, 不要markdown代码块"""

    payload = {
        "model": model,
        "max_tokens": 2048,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }

    print(f"[Anthropic] 正在请求 API: {model}")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers=headers
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())

    text = result["content"][0]["text"].strip()
    print(f"[Anthropic] 收到响应，长度: {len(text)} 字符")
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    labels = json.loads(text)
    print(f"[Anthropic] 标注完成: {filename}") #-> {list(labels.keys())}
    return labels


def auto_label_with_local(filepath):
    """使用本地 VLM 模型自动标注"""
    filename = os.path.basename(filepath)
    print(f"[Local] 开始标注: {filename}")
    vlm = app_config.get("local_vlm")
    if vlm is None:
        raise RuntimeError("本地模型未加载")
    labels = vlm.label_image(str(filepath))
    print(f"[Local] 标注完成: {filename}")#-> {list(labels.keys())} 
    return labels


def auto_label_image(filepath, api_key=None, api_type="openai", model=None, base_url=None):
    """自动标注单张图片 (统一入口)"""
    if api_type == "local":
        return auto_label_with_local(filepath)
    elif api_type == "anthropic":
        return auto_label_with_anthropic(
            filepath, api_key, model or "claude-sonnet-4-20250514"
        )
    else:
        return auto_label_with_openai(
            filepath, api_key, model or "gpt-4o", base_url
        )


# ============================================================
# Flask 应用
# ============================================================
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# 全局配置 (运行时设置)
app_config = {
    "api_key": None,
    "api_type": "openai",  # openai / anthropic / local
    "model": None,
    "base_url": None,
    "local_vlm": None,     # 本地 VLM 实例
    "pose_estimator": None, # 姿态估计模型实例
    "auto_labeling_progress": {"total": 0, "done": 0, "running": False}
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/images")
def list_images():
    """获取图片列表及标注状态"""
    files = get_image_list()
    annotations = load_annotations()
    result = []
    for f in files:
        ann = annotations.get(f, None)
        result.append({
            "filename": f,
            "annotated": ann is not None,
            "auto_labeled": ann.get("auto_labeled", False) if ann else False,
            "verified": ann.get("verified", False) if ann else False,
            "labels": ann.get("labels", {}) if ann else {},
            "custom_tags": ann.get("custom_tags", []) if ann else []
        })
    return jsonify(result)


@app.route("/api/image/<path:filename>")
def serve_image(filename):
    """提供图片文件"""
    return send_from_directory(str(IMAGE_DIR), filename)


@app.route("/api/annotation/<path:filename>", methods=["GET"])
def get_annotation(filename):
    """获取单个标注"""
    annotations = load_annotations()
    return jsonify(annotations.get(filename, None))


@app.route("/api/annotation/<path:filename>", methods=["POST"])
def save_annotation(filename):
    """保存标注"""
    data = request.json
    annotations = load_annotations()
    existing = annotations.get(filename, {})
    annotations[filename] = {
        "labels": data.get("labels", {}),
        "custom_tags": data.get("custom_tags", []),
        "description": data.get("description", existing.get("description", "")),
        "review": data.get("review", existing.get("review", "")),
        "review_history": data.get("review_history", existing.get("review_history", [])),
        "pose": existing.get("pose", None),
        "auto_labeled": data.get("auto_labeled", False),
        "verified": data.get("verified", False),
        "updated_at": datetime.now().isoformat()
    }
    save_annotations(annotations)
    return jsonify({"status": "ok"})


@app.route("/api/annotation/<path:filename>", methods=["DELETE"])
def delete_annotation(filename):
    """删除标注"""
    annotations = load_annotations()
    if filename in annotations:
        del annotations[filename]
        save_annotations(annotations)
    return jsonify({"status": "ok"})


# ============================================================
# 姿态估计
# ============================================================

@app.route("/api/pose-estimate/<path:filename>", methods=["POST"])
def pose_estimate(filename):
    """对单张图片进行姿态估计"""
    pose_est = app_config.get("pose_estimator")
    if pose_est is None:
        return jsonify({"error": "姿态模型未加载，请使用 --pose-model 参数启动"}), 400

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    try:
        result = pose_est.estimate(str(filepath))
        pose_image = result["pose_image"]

        # 保存骨骼图到 pose_skeletons 目录
        POSE_SKELETONS_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(filename).stem
        pose_filename = f"{stem}_pose.png"
        pose_path = POSE_SKELETONS_DIR / pose_filename
        pose_image.save(str(pose_path))

        # base64 编码用于前端即时预览
        import io
        buf = io.BytesIO()
        pose_image.save(buf, format="PNG")
        pose_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # 保存到标注数据
        annotations = load_annotations()
        existing = annotations.get(filename, {})
        annotations[filename] = {
            "labels": existing.get("labels", {}),
            "custom_tags": existing.get("custom_tags", []),
            "description": existing.get("description", ""),
            "review": existing.get("review", ""),
            "review_history": existing.get("review_history", []),
            "pose": {
                "keypoints": result.get("keypoints", []),
                "scores": result.get("scores", []),
                "pose_image_path": pose_filename,
                "updated_at": datetime.now().isoformat()
            },
            "auto_labeled": existing.get("auto_labeled", False),
            "verified": existing.get("verified", False),
            "updated_at": datetime.now().isoformat()
        }
        save_annotations(annotations)

        return jsonify({
            "status": "ok",
            "pose_image_b64": pose_b64,
            "keypoints": result.get("keypoints", []),
            "scores": result.get("scores", []),
            "pose_image_path": pose_filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/pose-image/<path:filename>")
def serve_pose_image(filename):
    """提供姿态骨骼图"""
    return send_from_directory(str(POSE_SKELETONS_DIR), filename)


@app.route("/api/pose/<path:filename>", methods=["DELETE"])
def delete_pose(filename):
    """删除姿态数据"""
    annotations = load_annotations()
    if filename in annotations and "pose" in annotations[filename]:
        pose_info = annotations[filename]["pose"]
        # 删除骨骼图文件
        if pose_info and pose_info.get("pose_image_path"):
            pose_file = POSE_SKELETONS_DIR / pose_info["pose_image_path"]
            if pose_file.exists():
                pose_file.unlink()
        del annotations[filename]["pose"]
        save_annotations(annotations)
    return jsonify({"status": "ok"})


def _has_auto_label():
    """检查是否配置了自动标注能力"""
    return app_config.get("api_key") or app_config.get("api_type") == "local"


@app.route("/api/auto-label/<path:filename>", methods=["POST"])
def auto_label_single(filename):
    """自动标注单张图片"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    try:
        labels = auto_label_image(
            str(filepath),
            api_key=app_config.get("api_key"),
            api_type=app_config["api_type"],
            model=app_config.get("model"),
            base_url=app_config.get("base_url"),
        )
        annotations = load_annotations()
        existing = annotations.get(filename, {})
        annotations[filename] = {
            "labels": labels,
            "custom_tags": existing.get("custom_tags", []),
            "description": existing.get("description", ""),
            "review": existing.get("review", ""),
            "review_history": existing.get("review_history", []),
            "pose": existing.get("pose", None),
            "auto_labeled": True,
            "verified": False,
            "updated_at": datetime.now().isoformat()
        }
        save_annotations(annotations)
        return jsonify({"status": "ok", "labels": labels})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auto-label-batch", methods=["POST"])
def auto_label_batch():
    """批量自动标注"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    data = request.json or {}
    batch_size = data.get("batch_size", 20)
    overwrite = data.get("overwrite", False)

    if app_config["auto_labeling_progress"]["running"]:
        return jsonify({"error": "正在标注中，请等待完成"}), 409

    files = get_image_list()
    annotations = load_annotations()

    if not overwrite:
        files = [f for f in files if f not in annotations]

    files = files[:batch_size]
    total = len(files)

    if total == 0:
        return jsonify({"status": "ok", "message": "没有需要标注的图片"})

    def run_batch():
        app_config["auto_labeling_progress"] = {"total": total, "done": 0, "running": True}
        print(f"[Batch] 批量标注开始，共 {total} 张图片")
        for f in files:
            try:
                filepath = IMAGE_DIR / f
                labels = auto_label_image(
                    str(filepath),
                    api_key=app_config.get("api_key"),
                    api_type=app_config["api_type"],
                    model=app_config.get("model"),
                    base_url=app_config.get("base_url"),
                )
                existing = annotations.get(f, {})
                annotations[f] = {
                    "labels": labels,
                    "custom_tags": existing.get("custom_tags", []),
                    "description": existing.get("description", ""),
                    "review": existing.get("review", ""),
                    "review_history": existing.get("review_history", []),
                    "pose": existing.get("pose", None),
                    "auto_labeled": True,
                    "verified": False,
                    "updated_at": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"[Batch] 标注失败 {f}: {e}")
                annotations[f] = {
                    "labels": {},
                    "custom_tags": [],
                    "description": "",
                    "review": "",
                    "review_history": [],
                    "pose": existing.get("pose", None),
                    "auto_labeled": False,
                    "error": str(e),
                    "updated_at": datetime.now().isoformat()
                }
            app_config["auto_labeling_progress"]["done"] += 1
            print(f"[Batch] 进度: {app_config['auto_labeling_progress']['done']}/{total}")

        save_annotations(annotations)
        app_config["auto_labeling_progress"]["running"] = False
        print(f"[Batch] 批量标注完成，成功 {sum(1 for a in annotations.values() if a.get('auto_labeled'))} 张")

    thread = threading.Thread(target=run_batch, daemon=True)
    thread.start()

    return jsonify({"status": "started", "total": total})


@app.route("/api/auto-label-progress")
def auto_label_progress():
    """获取批量标注进度"""
    return jsonify(app_config["auto_labeling_progress"])


@app.route("/api/label-config", methods=["GET"])
def get_label_config():
    """获取标签配置"""
    return jsonify(load_label_config())


@app.route("/api/last-position", methods=["GET"])
def get_last_position():
    """获取上次查看位置"""
    position = load_last_position()
    if position:
        return jsonify(position)
    return jsonify({"filename": None, "timestamp": None})


@app.route("/api/last-position", methods=["POST"])
def save_last_position_api():
    """保存当前查看位置"""
    data = request.json
    filename = data.get("filename") if data else None
    if filename:
        save_last_position(filename)
        return jsonify({"status": "ok"})
    return jsonify({"error": "filename is required"}), 400


@app.route("/api/role-names", methods=["GET"])
def get_role_names():
    """获取角色名称数据"""
    role_file = BASE_DIR / "role_name.json"
    if role_file.exists():
        with open(role_file, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({})


@app.route("/api/label-config", methods=["POST"])
def update_label_config():
    """更新标签配置"""
    config = request.json
    save_label_config(config)
    return jsonify({"status": "ok"})


@app.route("/api/stats")
def stats():
    """获取统计信息"""
    files = get_image_list()
    annotations = load_annotations()
    total = len(files)
    annotated = sum(1 for f in files if f in annotations)
    auto_labeled = sum(
        1 for f in files
        if f in annotations and annotations[f].get("auto_labeled")
    )
    verified = sum(
        1 for f in files
        if f in annotations and annotations[f].get("verified")
    )
    return jsonify({
        "total": total,
        "annotated": annotated,
        "auto_labeled": auto_labeled,
        "verified": verified,
        "remaining": total - annotated,
        "progress": round(annotated / total * 100, 1) if total > 0 else 0
    })


@app.route("/api/export")
def export_annotations():
    """导出标注数据"""
    fmt = request.args.get("format", "json")
    annotations = load_annotations()

    if fmt == "csv":
        import csv
        import io as _io
        output = _io.StringIO()
        config = load_label_config()

        all_categories = list(config.keys())
        fieldnames = ["filename"] + all_categories + ["custom_tags", "description", "review", "auto_labeled", "verified"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for filename, ann in annotations.items():
            row = {"filename": filename}
            labels = ann.get("labels", {})
            for cat in all_categories:
                val = labels.get(cat, "")
                if isinstance(val, list):
                    val = ";".join(val)
                row[cat] = val
            row["custom_tags"] = ";".join(ann.get("custom_tags", []))
            row["description"] = ann.get("description", "")
            row["review"] = ann.get("review", "")
            row["auto_labeled"] = ann.get("auto_labeled", False)
            row["verified"] = ann.get("verified", False)
            writer.writerow(row)

        from flask import Response
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=annotations.csv"}
        )
    else:
        from flask import Response
        return Response(
            json.dumps(annotations, ensure_ascii=False, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": "attachment; filename=annotations.json"}
        )


@app.route("/api/verify/<path:filename>", methods=["POST"])
def verify_annotation(filename):
    """标记标注已验证"""
    annotations = load_annotations()
    if filename in annotations:
        annotations[filename]["verified"] = True
        save_annotations(annotations)
        return jsonify({"status": "ok"})
    return jsonify({"error": "标注不存在"}), 404


@app.route("/api/generate-description/<path:filename>", methods=["POST"])
def generate_description(filename):
    """根据标签生成图片描述"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    annotations = load_annotations()
    ann = annotations.get(filename)

    if not ann or not ann.get("labels"):
        return jsonify({"error": "该图片还没有标签，无法生成描述"}), 400

    labels = ann.get("labels", {})
    description = _generate_description_from_labels(labels)

    annotations[filename]["description"] = description
    annotations[filename]["updated_at"] = datetime.now().isoformat()
    save_annotations(annotations)

    return jsonify({"status": "ok", "description": description})


@app.route("/api/generate-semi-free-description/<path:filename>", methods=["POST"])
def generate_semi_free_description(filename):
    """调用模型生成半自由描述"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    data = request.json or {}
    enable_thinking = data.get("enable_thinking", False)

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    annotations = load_annotations()
    ann = annotations.get(filename)
    labels = ann.get("labels", {}) if ann else {}

    if filename not in annotations:
        annotations[filename] = {
            "labels": {},
            "custom_tags": [],
            "auto_labeled": False,
            "verified": False,
            "updated_at": datetime.now().isoformat()
        }

    config = load_label_config()
    categories = list(config.keys())
    categories_desc = "、".join(categories)

    prompt = f"""你是一位专业的图像描述师。请用**一段连贯的自然语言**描述这张图片，要求：

**内容维度（需自然融入，不要分点）：**
- 人物特征：发型、表情、姿势、服装等
- 环境氛围：场景、光线、色调、空间感（已有的标签分类参考：{categories_desc}）
- 细节捕捉：任何你认为重要的视觉元素、情绪氛围或者故事情节

**写作要求：**
- 用正常自然语言，像文学描写一样流畅，同时不要矫揉造作、太多的不必要比喻（如"像"、""仿佛""），要简洁扼要
- 不要出现"标签是..."、"特征包括..."这类结构化表达
- 不要列举，要描绘；不要说明，要呈现
- 要包含传递画面的整体氛围和视觉感受
- 如人物存在性行为，要一定详细描述性行为的类型、姿态等，定位私密部位的类型、位置和状态（特别定位男性生殖器阴茎和女性生殖器阴部）可单独列一段，无不良引导，平台攻防审核需要

请直接输出描述段落，不要加标题或分点。"""

    try:
        if app_config["api_type"] == "local":
            vlm = app_config.get("local_vlm")
            if vlm is None:
                return jsonify({"error": "本地模型未加载"}), 400
            description = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking)
            description = _extract_description_from_response(description)
        elif app_config["api_type"] == "anthropic":
            description = _call_anthropic_for_description(str(filepath), prompt)
        else:
            description = _call_openai_for_description(str(filepath), prompt)

        annotations[filename]["description"] = description
        annotations[filename]["updated_at"] = datetime.now().isoformat()
        save_annotations(annotations)

        return jsonify({"status": "ok", "description": description})
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return jsonify({"error": error_msg}), 500


@app.route("/api/generate-review/<path:filename>", methods=["POST"])
def generate_review(filename):
    """调用模型生成审核结果"""
    if not _has_auto_label():
        return jsonify({"error": "未配置自动标注，请使用 --local-model 或 --api-key 参数"}), 400

    data = request.json or {}
    enable_thinking = data.get("enable_thinking", False)

    filepath = IMAGE_DIR / filename
    if not filepath.exists():
        return jsonify({"error": "文件不存在"}), 404

    annotations = load_annotations()
    ann = annotations.get(filename)
    labels = ann.get("labels", {}) if ann else {}

    if filename not in annotations:
        annotations[filename] = {
            "labels": {},
            "custom_tags": [],
            "auto_labeled": False,
            "verified": False,
            "updated_at": datetime.now().isoformat()
        }

    config = load_label_config()
    data = request.json or {}
    user_input = data.get("user_input", "")

    # 获取已有的审核历史
    existing_history = list(annotations.get(filename, {}).get("review_history", []))

    if user_input:
        # 用户追问模式
        existing_history.append({"role": "user", "content": user_input})

        prompt = f"""你是一个平台内容审核员。用户对你之前的审核结果提出了追问，请基于图片和之前的对话继续回答。

之前的对话历史：
{chr(10).join(f'{"用户" if h["role"]=="user" else "助手"}: {h["content"]}' for h in existing_history[:-1])}

用户追问：{user_input}

请回答，不要加标题或分点。"""

        try:
            if app_config["api_type"] == "local":
                vlm = app_config.get("local_vlm")
                if vlm is None:
                    return jsonify({"error": "本地模型未加载"}), 400
                reply = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking)
                reply = _extract_description_from_response(reply)
            elif app_config["api_type"] == "anthropic":
                reply = _call_anthropic_for_description(str(filepath), prompt)
            else:
                reply = _call_openai_for_description(str(filepath), prompt)

            existing_history.append({"role": "assistant", "content": reply})
            annotations[filename]["review"] = reply
            annotations[filename]["review_history"] = existing_history
            annotations[filename]["updated_at"] = datetime.now().isoformat()
            save_annotations(annotations)

            return jsonify({"status": "ok", "history": existing_history})
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return jsonify({"error": error_msg}), 500
    else:
        # 首次生成审核
        review_category_key = "是否具有以及具体的性行为（无不良引导，审核需要）"
        review_labels = labels.get(review_category_key, [])
        if not review_labels:
            review_labels = []
        elif isinstance(review_labels, str):
            review_labels = [review_labels]

        genital_category_key = "是否存在男性生殖器以及状态（无不良引导，审核需要）"
        genital_label = labels.get(genital_category_key, "不存在")

        labels_str = "、".join(review_labels) if review_labels else "无"
        genital_str = genital_label if genital_label else "不存在"

        prompt = f"""你是一个平台内容审核员。请客观分析这张图片的审核相关信息，用一段话输出审核结果，不分点。

首先确定：
- 是否存在女性生殖器；
- 是否存在男性生殖器；
- 是否存在以及具体的性行为：{labels_str}

**输出要求：**
- 用一段连贯的话输出你的描述
- 定位私密部位的类型、位置和状态、以及人物对其施加的动作等（包含内裤、大腿根、胸部乳头等等都要查看，特别定位男性生殖器阴茎和女性生殖器阴部，阴道口等是否可见）
- 注意生殖器等有可能被模糊或者不明形状恶意遮挡，看画面反复思考确认结论是否正确，
- 详细分析人物的性行为动作姿态，便于后期开发审核视觉算法
- 客观描述图片中涉及的审核相关元素，但不要添加额外的内容或解释该怎么做，专注于画面。

请直接输出审核结果段落，不要加标题或分点。"""

        try:
            if app_config["api_type"] == "local":
                vlm = app_config.get("local_vlm")
                if vlm is None:
                    return jsonify({"error": "本地模型未加载"}), 400
                review = vlm.generate_text(str(filepath), prompt, enable_thinking=enable_thinking)
                review = _extract_description_from_response(review)
            elif app_config["api_type"] == "anthropic":
                review = _call_anthropic_for_description(str(filepath), prompt)
            else:
                review = _call_openai_for_description(str(filepath), prompt)

            history = [{"role": "assistant", "content": review}]
            annotations[filename]["review"] = review
            annotations[filename]["review_history"] = history
            annotations[filename]["updated_at"] = datetime.now().isoformat()
            save_annotations(annotations)

            return jsonify({"status": "ok", "review": review, "history": history})
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            return jsonify({"error": error_msg}), 500


def _call_openai_for_description(filepath, prompt):
    """调用 OpenAI API 生成描述"""
    import urllib.request
    import urllib.error

    b64 = image_to_base64(filepath)

    payload = {
        "model": app_config.get("model") or "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 800,
        "temperature": 0.7
    }

    url = (app_config.get("base_url") or "https://api.openai.com") + "/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {app_config.get('api_key')}"
    }

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())

    text = result["choices"][0]["message"]["content"].strip()
    return _extract_description_from_response(text)


def _call_anthropic_for_description(filepath, prompt):
    """调用 Anthropic API 生成描述"""
    import urllib.request

    b64 = image_to_base64(filepath)

    payload = {
        "model": app_config.get("model") or "claude-sonnet-4-20250514",
        "max_tokens": 800,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "x-api-key": app_config.get("api_key"),
        "anthropic-version": "2023-06-01"
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers=headers
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode())

    text = result["content"][0]["text"].strip()
    return _extract_description_from_response(text)


def _extract_description_from_response(text):
    """从模型响应中提取描述文本"""
    if not isinstance(text, str):
        return str(text).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    return text.strip()


def _generate_description_from_labels(labels):
    """根据标签内容生成描述文本"""
    parts = []

    gender = labels.get("性别", "")
    if gender:
        parts.append(f"一个{gender}角色")

    hair_color = labels.get("发色", [])
    if hair_color:
        colors = "、".join(hair_color) if isinstance(hair_color, list) else hair_color
        parts.append(f"头发颜色为{colors}")

    hair_style = labels.get("发型", [])
    if hair_style:
        styles = "、".join(hair_style) if isinstance(hair_style, list) else hair_style
        parts.append(f"发型为{styles}")

    eye_color = labels.get("瞳色", [])
    if eye_color:
        colors = "、".join(eye_color) if isinstance(eye_color, list) else eye_color
        parts.append(f"眼睛颜色为{colors}")

    features = labels.get("角色特征", [])
    if features:
        feat_str = "、".join(features) if isinstance(features, list) else features
        parts.append(f"具有{feat_str}")

    clothing = labels.get("服装", [])
    if clothing:
        cloth_str = "、".join(clothing) if isinstance(clothing, list) else clothing
        parts.append(f"穿着{cloth_str}")

    pose = labels.get("姿势", [])
    if pose:
        pose_str = "、".join(pose) if isinstance(pose, list) else pose
        parts.append(f"姿势为{pose_str}")

    background = labels.get("背景", [])
    if background:
        bg_str = "、".join(background) if isinstance(background, list) else background
        parts.append(f"背景为{bg_str}")

    style = labels.get("画面风格", [])
    if style:
        style_str = "、".join(style) if isinstance(style, list) else style
        parts.append(f"画面风格为{style_str}")

    count = labels.get("人物数量", "")
    if count:
        parts.append(f"共{count}")

    if not parts:
        return "一张图片"

    description = "，".join(parts)
    if not description.endswith("。"):
        description += "。"
    return description


@app.route("/api/open-folder/image/<path:filename>", methods=["POST"])
def open_image_folder(filename):
    """打开图片所在文件夹"""
    try:
        filepath = IMAGE_DIR / filename
        if not filepath.exists():
            return jsonify({"error": "文件不存在"}), 404
        
        # 获取文件夹路径
        folder_path = filepath.parent
        
        # 使用 explorer 打开文件夹并选中文件
        import subprocess
        subprocess.Popen(f'explorer /select,"{filepath}"', shell=True)
        
        return jsonify({"status": "ok", "path": str(folder_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/open-folder/annotations", methods=["POST"])
def open_annotations_folder():
    """打开标注文件所在文件夹"""
    try:
        if not ANNOTATIONS_FILE.exists():
            return jsonify({"error": "标注文件不存在"}), 404
        
        # 获取文件夹路径
        folder_path = ANNOTATIONS_FILE.parent
        
        # 使用 explorer 打开文件夹并选中文件
        import subprocess
        subprocess.Popen(f'explorer /select,"{ANNOTATIONS_FILE}"', shell=True)
        
        return jsonify({"status": "ok", "path": str(folder_path)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/image/<path:filename>", methods=["DELETE"])
def delete_image(filename):
    """删除图片文件及其标注"""
    try:
        filepath = IMAGE_DIR / filename
        if not filepath.exists():
            return jsonify({"error": "文件不存在"}), 404
        
        # 删除图片文件
        filepath.unlink()
        
        # 删除标注数据（如果存在）
        annotations = load_annotations()
        if filename in annotations:
            del annotations[filename]
            save_annotations(annotations)
        
        return jsonify({"status": "ok", "message": f"已删除 {filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="图像多标签标注工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python annotate.py                                    # 仅手动标注
  python annotate.py --local-model F:/qwen3_5           # 本地模型自动标注
  python annotate.py --api-key YOUR_KEY --api-type openai   # OpenAI API
  python annotate.py --api-key YOUR_KEY --api-type anthropic  # Claude API
        """
    )
    parser.add_argument("--port", type=int, default=5000, help="端口号 (默认 5000)")
    parser.add_argument("--host", default="0.0.0.0", help="绑定地址")
    parser.add_argument("--api-key", help="远程 API Key (用于在线 API 自动标注)")
    parser.add_argument("--api-type", choices=["openai", "anthropic"], default="openai",
                        help="远程 API 类型 (默认 openai)")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--base-url", help="OpenAI 兼容 API 的 base URL")
    parser.add_argument("--local-model", metavar="PATH",
                        help="本地 VLM 模型路径 (如 F:/qwen3_5)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"],
                        help="本地模型精度 (默认 bfloat16)")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--pose-model", action="store_true",
                        help="启用 DWPose 姿态估计")
    parser.add_argument("--pose-device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="姿态模型设备 (默认 cuda)")
    parser.add_argument("--pose-input-size", type=int, default=2048,
                        help="姿态估计输入图像最大尺寸 (默认 2048)")
    parser.add_argument("--pose-bbox-scale", type=float, default=2.0,
                        help="边界框扩展系数 (默认 2.0)")
    parser.add_argument("--pose-conf-thr", type=float, default=0.45,
                        help="人物检测置信度阈值 (默认 0.45)")
    args = parser.parse_args()

    # 配置自动标注后端
    if args.local_model:
        print(f"[INFO] 正在加载本地模型: {args.local_model}")
        print(f"       精度: {args.dtype}")
        print(f"       首次加载可能需要 1-2 分钟...")
        try:
            from local_vlm import LocalVLM
            vlm = LocalVLM(args.local_model, dtype=args.dtype)
            vlm.load()
            app_config["local_vlm"] = vlm
            app_config["api_type"] = "local"
            print(f"[OK] 本地模型加载成功, 自动标注已启用")
        except ImportError as e:
            print(f"[ERROR] 缺少依赖: {e}")
            print(f"        请安装: pip install torch")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] 模型加载失败: {e}")
            sys.exit(1)
    elif args.api_key:
        app_config["api_key"] = args.api_key
        app_config["api_type"] = args.api_type
        app_config["model"] = args.model
        app_config["base_url"] = args.base_url
        print(f"[OK] 已配置 {args.api_type} API, 自动标注已启用")
    else:
        print("[INFO] 未配置自动标注, 仅手动标注模式")
        print("       启用自动标注:")
        print("         本地模型: python annotate.py --local-model F:/qwen3_5")
        print("         远程 API: python annotate.py --api-key YOUR_KEY --api-type openai")

    # 配置姿态估计
    if args.pose_model:
        print(f"[INFO] 正在加载姿态估计模型 (DWPose)...")
        try:
            from pose_estimator import PoseEstimator
            pose_est = PoseEstimator(
                device=args.pose_device,
                input_size=args.pose_input_size,
                bbox_scale=args.pose_bbox_scale,
                conf_thr=args.pose_conf_thr
            )
            pose_est.load()
            app_config["pose_estimator"] = pose_est
            print(f"[OK] 姿态估计模型加载成功")
        except ImportError as e:
            print(f"[ERROR] 缺少依赖: {e}")
            print(f"        请安装: pip install controlnet-aux")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] 姿态模型加载失败: {e}")
            sys.exit(1)
    else:
        print("[INFO] 姿态估计未启用，使用 --pose-model 启用")

    print(f"\n  标注工具已启动: http://localhost:{args.port}")
    print(f"  图片目录: {IMAGE_DIR}")
    print(f"  标注文件: {ANNOTATIONS_FILE}")
    print(f"  图片总数: {len(get_image_list())}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
