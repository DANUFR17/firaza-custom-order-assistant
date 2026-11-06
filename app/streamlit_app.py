"""
streamlit_app.py - Web admin Firaza Custom Order Management (Streamlit).

Web ini TIDAK menyentuh database atau model secara langsung: semua lewat REST API (FastAPI).

Jalankan (dua terminal, dari ROOT proyek, venv aktif):
    uvicorn api:app --app-dir src --reload          # terminal 1: API  (port 8000)
    streamlit run app/streamlit_app.py              # terminal 2: web  (port 8501)

Alamat API bisa diubah lewat variabel lingkungan API_URL (default http://127.0.0.1:8000).
"""
import hashlib
import html
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
WIB = timedelta(hours=7)          # database menyimpan UTC; ditampilkan sebagai WIB
STATUS_ICON = {"New": "🆕", "Design/Preparation": "🎨", "Production": "🏭", "Completed": "✅"}
NONE_CHOICE = "(belum ditentukan)"

COLOR_MAP = {"navy": "#1f2f5c", "hitam": "#222222", "maroon": "#7b1e2b", "hijau botol": "#1e5631", "abu-abu": "#8c8c8c",
             "abu": "#9a9a9a", "krem": "#f2e6c9", "biru dongker": "#1b2a49", "coklat": "#6b4226", "cokelat": "#6b4226",
             "ungu": "#5b3a7c", "tosca": "#2aa198", "putih": "#f5f5f5", "pink": "#f4a6c0", "hijau sage": "#9caf88",
             "biru muda": "#a7c7e7", "merah": "#b22222"}
FONT_CSS = {"Kaligrafi": "'Palatino Linotype', Georgia, serif; font-style:italic", "Script": "'Brush Script MT', cursive",
            "Arab": "'Traditional Arabic', serif", "Elegan": "Georgia, serif", "Klasik": "'Times New Roman', serif",
            "Sans Simple": "Arial, sans-serif", "Bold": "'Arial Black', Arial, sans-serif; font-weight:900",
            "Handwriting": "'Comic Sans MS', cursive"}
LAYOUT_ALIGN = {"Tengah": ("center", "center"), "Atas": ("flex-start", "center"), "Bawah": ("flex-end", "center"),
                "Pojok Kanan Bawah": ("flex-end", "flex-end"), "Pojok Kiri Atas": ("flex-start", "flex-start")}


# ================================================================== util API
def call(method, path, ok=(200, 201, 204), **kw):
    """Panggil API. Return Response bila sukses; tampilkan pesan galat dan return None bila gagal."""
    try:
        r = requests.request(method, API + path, timeout=60, **kw)
    except requests.exceptions.RequestException:
        st.error(f"API tidak dapat dihubungi di {API}. Jalankan dulu: `uvicorn api:app --app-dir src --reload`")
        st.stop()
    if r.status_code in ok:
        return r
    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    if isinstance(detail, dict):
        detail = detail.get("message", str(detail)) + (f" Transisi diizinkan: {detail['transisi_diizinkan']}" if "transisi_diizinkan" in detail else "")
    elif isinstance(detail, list):
        detail = "; ".join(f"{'.'.join(str(x) for x in d.get('loc', []))}: {d.get('msg')}" for d in detail)
    st.error(f"Gagal ({r.status_code}): {detail}")
    return None


def get(path, **params):
    r = call("GET", path, params={k: v for k, v in params.items() if v not in (None, "")})
    return r.json() if r is not None else None


def fmt_dt(s):
    if not s:
        return "-"
    return (datetime.fromisoformat(str(s).replace("Z", "")) + WIB).strftime("%d %b %Y %H:%M WIB")


@st.cache_data(ttl=60, show_spinner=False)
def load_options():
    return requests.get(API + "/options", timeout=30).json()


def clean(v):
    return None if (v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "") else str(v).strip()


def preview_html(item):
    """Pratinjau KASAR posisi dan gaya teks di atas warna produk (bukan hasil cetak)."""
    bg = COLOR_MAP.get((item.get("color") or "").lower(), "#d9d9d9")
    r, g, b = int(bg[1:3], 16), int(bg[3:5], 16), int(bg[5:7], 16)
    fg = "#ffffff" if (0.299 * r + 0.587 * g + 0.114 * b) < 140 else "#222222"
    align_items, justify = LAYOUT_ALIGN.get(item.get("layout") or "Tengah", ("center", "center"))
    font = FONT_CSS.get(item.get("font") or "", "Arial, sans-serif")
    text = html.escape(item.get("custom_text") or "")
    return (f"<div style='background:{bg};color:{fg};height:110px;border-radius:10px;padding:10px;display:flex;"
            f"align-items:{align_items};justify-content:{justify};border:1px solid #bbb;'>"
            f"<span style='font-family:{font};font-size:26px;'>{text}</span></div>")


def products_map():
    prods = get("/products") or []
    return {p["name"]: p["id"] for p in prods}, {p["id"]: p["name"] for p in prods}


def customers_map():
    cs = get("/customers", limit=200) or []
    return {f"{c['name']} (#{c['id']})": c["id"] for c in cs}, {c["id"]: c["name"] for c in cs}


# ================================================================== halaman
def page_dashboard():
    st.header("Dashboard")
    orders = get("/orders", limit=200) or []
    cols = st.columns(4)
    for col, s in zip(cols, load_options()["statuses"]):
        col.metric(f"{STATUS_ICON[s]} {s}", sum(1 for o in orders if o["status"] == s))
    if not orders:
        st.info("Belum ada order. Buat di menu **Order Baru**.")
        return
    _, cname = customers_map()
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Jenis order")
        counts = pd.Series([o["order_type"] or o["predicted_type"] or "Belum ditentukan" for o in orders]).value_counts()
        st.bar_chart(counts)
    with right:
        st.subheader("Order terbaru")
        st.dataframe(pd.DataFrame([{"Kode": o["order_code"], "Customer": cname.get(o["customer_id"], "?"),
                                    "Jenis": o["order_type"] or (f"{o['predicted_type']} (usulan AI)" if o["predicted_type"] else "-"),
                                    "Status": f"{STATUS_ICON[o['status']]} {o['status']}", "Dibuat": fmt_dt(o["created_at"])}
                                   for o in orders[:10]]), hide_index=True)


def page_order_baru():
    st.header("Order Baru")
    opts = load_options()
    pname, _ = products_map()
    cmap, _ = customers_map()

    st.subheader("1. Customer")
    mode = st.radio("Customer", ["Pilih yang sudah ada", "Customer baru"], horizontal=True, label_visibility="collapsed", key="cust_mode")
    customer_id, new_customer = None, None
    if mode == "Pilih yang sudah ada":
        if cmap:
            customer_id = cmap[st.selectbox("Pilih customer", list(cmap), key="cust_pick")]
        else:
            st.info("Belum ada customer, pilih **Customer baru**.")
    else:
        c1, c2 = st.columns(2)
        new_customer = {"name": c1.text_input("Nama", key="nc_name"), "phone": c2.text_input("Telepon/WA", key="nc_phone"),
                        "channel": c1.selectbox("Kanal", opts["channels"], key="nc_channel"),
                        "address": c2.text_input("Alamat", key="nc_addr")}

    st.subheader("2. Pesan customer dan analisis AI")
    raw = st.text_area("Pesan customer", height=110, key="raw_msg",
                       placeholder="Contoh: Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah")
    if st.button("Analisis dengan AI", key="btn_classify", disabled=not raw.strip()):
        r = call("POST", "/classify", json={"text": raw})
        if r is not None:
            st.session_state["pred"] = {"text": raw, **r.json()}
            st.session_state["order_type_pick"] = r.json()["label"]   # pilihan jenis order mengikuti usulan AI
    pred = st.session_state.get("pred")
    if pred and pred["text"] == raw:
        a, b = st.columns([1, 2])
        a.metric("Usulan jenis order", pred["label"], f"confidence {pred['confidence']:.0%}", delta_color="off")
        if pred["needs_review"]:
            a.warning("Confidence rendah, mohon periksa manual.")
        b.bar_chart(pd.Series(pred["probabilities"]))
    elif pred:
        st.caption("Pesan berubah sejak analisis terakhir, klik Analisis lagi.")

    st.subheader("3. Detail order")
    choices = [NONE_CHOICE] + opts["order_types"]
    order_type = st.selectbox("Jenis order (konfirmasi admin)", choices, key="order_type_pick",
                              help="Usulan AI hanya saran (terisi otomatis setelah analisis); admin yang menentukan jenis akhir.")
    c1, c2 = st.columns(2)
    channel = c1.selectbox("Kanal order", opts["channels"], key="order_channel")
    has_due = c2.checkbox("Ada tenggat", key="has_due")
    due = c2.date_input("Tenggat", key="due_date", disabled=not has_due)
    notes = st.text_input("Catatan", key="order_notes")

    st.markdown("**Item pesanan** (tambah baris dengan tombol + di bawah tabel)")
    editor_df = pd.DataFrame({"Produk": pd.Series([], dtype="object"), "Jumlah": pd.Series([], dtype="int64"),
                              "Warna": pd.Series([], dtype="object"), "Nama/teks custom": pd.Series([], dtype="object"),
                              "Font": pd.Series([], dtype="object"), "Layout": pd.Series([], dtype="object")})
    items_df = st.data_editor(editor_df, num_rows="dynamic", key="items_editor", hide_index=True, column_config={
        "Produk": st.column_config.SelectboxColumn(options=list(pname)),
        "Jumlah": st.column_config.NumberColumn(min_value=1, step=1, default=1),
        "Font": st.column_config.SelectboxColumn(options=opts["fonts"]),
        "Layout": st.column_config.SelectboxColumn(options=opts["layouts"])})

    if st.button("Simpan Order", type="primary", key="btn_save_order"):
        if mode == "Customer baru":
            if not (new_customer["name"] or "").strip():
                st.error("Nama customer wajib diisi.")
                return
            r = call("POST", "/customers", json={k: (v or None) for k, v in new_customer.items()})
            if r is None:
                return
            customer_id = r.json()["id"]
        if customer_id is None:
            st.error("Pilih atau buat customer dulu.")
            return
        items = []
        for _, row in items_df.iterrows():
            prod, text = clean(row.get("Produk")), clean(row.get("Nama/teks custom"))
            if prod is None and text is None:
                continue
            qty = row.get("Jumlah")
            items.append({"product_id": pname.get(prod) if prod else None, "quantity": int(qty) if pd.notna(qty) else 1,
                          "color": clean(row.get("Warna")), "custom_text": text, "font": clean(row.get("Font")),
                          "layout": clean(row.get("Layout"))})
        payload = {"customer_id": customer_id, "raw_message": raw.strip() or None, "channel": channel,
                   "auto_classify": bool(raw.strip()), "notes": notes or None, "items": items}
        if order_type != NONE_CHOICE:
            payload["order_type"] = order_type
        if has_due:
            payload["due_date"] = datetime.combine(due, datetime.min.time()).isoformat()
        r = call("POST", "/orders", json=payload)
        if r is not None:
            o = r.json()
            st.session_state["last_order"] = o["order_code"]
            st.success(f"Order **{o['order_code']}** tersimpan (status {o['status']}). Buka menu **Daftar Order** untuk membuat "
                       "ringkasan AI, kartu ucapan, dan mengubah status.")


def render_summary(s):
    st.markdown(f"**Jenis order:** {s.get('jenis_order') or '-'}  ·  **Sumber:** "
                f"{'Gemini' if s.get('sumber') == 'gemini' else 'Fallback berbasis aturan'}")
    st.write(s.get("ringkasan", ""))
    if s.get("items"):
        st.dataframe(pd.DataFrame(s["items"]).rename(columns={"produk": "Produk", "jumlah": "Jumlah", "warna": "Warna",
                                                              "teks_custom": "Teks custom", "font": "Font", "layout": "Layout"}),
                     hide_index=True)
    st.write(f"**Acara:** {s.get('acara') or '-'}  ·  **Tenggat disebut:** {s.get('tenggat') or '-'}")
    for q in s.get("perlu_klarifikasi", []):
        st.warning(f"Perlu klarifikasi: {q}")
    for w in s.get("peringatan", []):
        st.caption(f"⚠️ {w}")


def page_daftar_order():
    st.header("Daftar Order")
    opts = load_options()
    pname, pid_name = products_map()
    _, cname = customers_map()

    f1, f2, f3 = st.columns(3)
    status = f1.selectbox("Status", ["Semua"] + opts["statuses"], key="f_status")
    jenis = f2.selectbox("Jenis", ["Semua"] + opts["order_types"], key="f_jenis")
    q = f3.text_input("Cari kode/pesan", key="f_q")
    orders = get("/orders", status=None if status == "Semua" else status, order_type=None if jenis == "Semua" else jenis,
                 q=q, limit=200) or []
    if not orders:
        st.info("Tidak ada order yang cocok.")
        return
    st.dataframe(pd.DataFrame([{"Kode": o["order_code"], "Customer": cname.get(o["customer_id"], "?"),
                                "Jenis": o["order_type"] or (f"{o['predicted_type']} (usulan)" if o["predicted_type"] else "-"),
                                "Status": f"{STATUS_ICON[o['status']]} {o['status']}", "Item": len(o["items"]),
                                "Tenggat": fmt_dt(o["due_date"]) if o["due_date"] else "-", "Dibuat": fmt_dt(o["created_at"])}
                               for o in orders]), hide_index=True)

    labels = {f"{o['order_code']} · {cname.get(o['customer_id'], '?')} · {o['status']}": o["id"] for o in orders}
    preselect = st.session_state.get("last_order")
    default = next((i for i, l in enumerate(labels) if l.startswith(str(preselect))), 0)
    oid = labels[st.selectbox("Pilih order untuk dikelola", list(labels), index=default, key="order_pick")]
    o = get(f"/orders/{oid}")
    if o is None:
        return
    st.divider()
    st.subheader(f"{o['order_code']}  {STATUS_ICON[o['status']]} {o['status']}")
    tabs = st.tabs(["Detail", "Item", "Ringkasan AI", "Kartu Ucapan", "Status & Riwayat", "Edit / Hapus"])

    with tabs[0]:
        st.write(f"**Customer:** {cname.get(o['customer_id'], '?')}  ·  **Kanal:** {o['channel']}  ·  **Dibuat:** {fmt_dt(o['created_at'])}")
        st.write(f"**Tenggat:** {fmt_dt(o['due_date']) if o['due_date'] else '-'}  ·  **Catatan:** {o['notes'] or '-'}")
        st.markdown("**Pesan asli customer**")
        st.info(o["raw_message"] or "(tidak ada)")
        pc = f" (confidence {o['predicted_confidence']:.0%})" if o["predicted_confidence"] is not None else ""
        st.write(f"**Usulan AI:** {o['predicted_type'] or '-'}{pc}  ·  **Jenis final:** {o['order_type'] or 'belum dikonfirmasi'}")
        if o["predicted_type"] and not o["order_type"]:
            if st.button(f"Konfirmasi jenis: {o['predicted_type']}", key="btn_confirm_type"):
                if call("PUT", f"/orders/{oid}", json={"order_type": o["predicted_type"]}) is not None:
                    st.rerun()

    with tabs[1]:
        if o["items"]:
            st.dataframe(pd.DataFrame([{"ID": i["id"], "Produk": pid_name.get(i["product_id"], "-"), "Jumlah": i["quantity"],
                                        "Warna": i["color"], "Teks custom": i["custom_text"], "Font": i["font"], "Layout": i["layout"]}
                                       for i in o["items"]]), hide_index=True)
            for i in o["items"]:
                if i["custom_text"]:
                    st.caption(f"Pratinjau kasar item #{i['id']} (posisi dan gaya teks, bukan hasil cetak)")
                    st.markdown(preview_html(i), unsafe_allow_html=True)
        else:
            st.info("Belum ada item.")
        with st.form("form_add_item", clear_on_submit=True):
            st.markdown("**Tambah item**")
            a, b, c = st.columns(3)
            prod = a.selectbox("Produk", [""] + list(pname), key="ai_prod")
            qty = b.number_input("Jumlah", min_value=1, value=1, step=1, key="ai_qty")
            color = c.text_input("Warna", key="ai_color")
            text = a.text_input("Nama/teks custom", key="ai_text")
            font = b.selectbox("Font", [""] + opts["fonts"], key="ai_font")
            layout = c.selectbox("Layout", [""] + opts["layouts"], key="ai_layout")
            if st.form_submit_button("Tambah item"):
                payload = {"product_id": pname.get(prod) if prod else None, "quantity": int(qty), "color": color or None,
                           "custom_text": text or None, "font": font or None, "layout": layout or None}
                if call("POST", f"/orders/{oid}/items", json=payload) is not None:
                    st.rerun()
        if o["items"]:
            d1, d2 = st.columns([2, 1])
            del_id = d1.selectbox("Hapus item", [i["id"] for i in o["items"]], key="del_item_id")
            if d2.button("Hapus", key="btn_del_item"):
                if call("DELETE", f"/orders/{oid}/items/{del_id}") is not None:
                    st.rerun()

    with tabs[2]:
        if o["ai_summary"]:
            render_summary(o["ai_summary"])
        else:
            st.info("Belum ada ringkasan. Catat item order dulu agar nama dan jumlah akurat, lalu buat ringkasan.")
        b1, b2 = st.columns(2)
        if b1.button("Buat ringkasan", key="btn_sum", disabled=bool(o["ai_summary"])):
            if call("POST", f"/orders/{oid}/summary") is not None:
                st.rerun()
        if b2.button("Buat ulang", key="btn_sum_force", disabled=not o["ai_summary"]):
            if call("POST", f"/orders/{oid}/summary", params={"force": "true"}) is not None:
                st.rerun()

    with tabs[3]:
        tone = st.selectbox("Nada", ["hangat", "formal", "islami"], key="tone")
        g1, g2 = st.columns(2)
        if g1.button("Buat draf kartu ucapan", key="btn_greet"):
            r = call("POST", f"/orders/{oid}/greeting", json={"tone": tone}, params={"force": "true"})
            if r is not None:
                st.session_state[f"greet_meta_{oid}"] = r.json()
                st.rerun()
        g2.caption("Draf hanya saran; periksa nama dan isi sebelum dicetak.")
        meta = st.session_state.get(f"greet_meta_{oid}")
        if meta:
            st.caption(f"Sumber draf terakhir: {'Gemini' if meta['sumber'] == 'gemini' else 'template (fallback)'}"
                       + (" · " + "; ".join(meta["peringatan"]) if meta.get("peringatan") else ""))
        # kunci memuat hash isi tersimpan: widget dengan key tetap mengabaikan `value` baru, sehingga draf hasil AI tidak akan tampil
        gkey = hashlib.md5((o["greeting_card"] or "").encode()).hexdigest()[:8]
        text = st.text_area("Teks kartu ucapan (boleh diedit)", value=o["greeting_card"] or "", height=130, key=f"greet_text_{oid}_{gkey}")
        if st.button("Simpan teks", key="btn_save_greet", disabled=not text.strip()):
            if call("PUT", f"/orders/{oid}", json={"greeting_card": text}) is not None:
                st.success("Tersimpan.")

    with tabs[4]:
        nexts = opts["status_transitions"].get(o["status"], [])
        if nexts:
            note = st.text_input("Catatan perubahan status (opsional)", key="status_note")
            cols = st.columns(len(nexts))
            for col, s in zip(cols, nexts):
                label = f"Kembali ke {s} (revisi)" if opts["statuses"].index(s) < opts["statuses"].index(o["status"]) else f"Pindah ke {s}"
                if col.button(f"{STATUS_ICON[s]} {label}", key=f"btn_status_{s}"):
                    if call("POST", f"/orders/{oid}/status", json={"status": s, "note": note or None}) is not None:
                        st.rerun()
        else:
            st.success("Order sudah selesai (Completed).")
        hist = get(f"/orders/{oid}/history") or []
        st.markdown("**Riwayat status**")
        st.dataframe(pd.DataFrame([{"Waktu": fmt_dt(h["changed_at"]), "Dari": h["from_status"] or "-", "Ke": h["to_status"],
                                    "Catatan": h["note"] or ""} for h in hist]), hide_index=True)

    with tabs[5]:
        choices = [NONE_CHOICE] + opts["order_types"]
        new_type = st.selectbox("Jenis order final", choices, key=f"edit_type_{oid}",
                                index=choices.index(o["order_type"]) if o["order_type"] in choices else 0)
        new_notes = st.text_input("Catatan", value=o["notes"] or "", key=f"edit_notes_{oid}")
        if st.button("Simpan perubahan", key="btn_edit_order"):
            payload = {"notes": new_notes}
            if new_type != NONE_CHOICE:
                payload["order_type"] = new_type
            if call("PUT", f"/orders/{oid}", json=payload) is not None:
                st.rerun()
        st.divider()
        sure = st.checkbox("Saya yakin ingin menghapus order ini", key="chk_delete")
        if st.button("Hapus order", key="btn_delete_order", disabled=not sure):
            if call("DELETE", f"/orders/{oid}") is not None:
                st.session_state.pop("last_order", None)
                st.rerun()


def page_customer():
    st.header("Customer")
    opts = load_options()
    cs = get("/customers", limit=200) or []
    if cs:
        st.dataframe(pd.DataFrame([{"ID": c["id"], "Nama": c["name"], "Telepon": c["phone"], "Kanal": c["channel"],
                                    "Alamat": c["address"], "Catatan": c["notes"], "Dibuat": fmt_dt(c["created_at"])} for c in cs]),
                     hide_index=True)
    with st.form("form_add_customer", clear_on_submit=True):
        st.markdown("**Tambah customer**")
        a, b = st.columns(2)
        name, phone = a.text_input("Nama", key="ac_name"), b.text_input("Telepon/WA", key="ac_phone")
        channel, address = a.selectbox("Kanal", opts["channels"], key="ac_channel"), b.text_input("Alamat", key="ac_addr")
        notes = st.text_input("Catatan", key="ac_notes")
        if st.form_submit_button("Simpan"):
            if not name.strip():
                st.error("Nama wajib diisi.")
            elif call("POST", "/customers", json={"name": name, "phone": phone or None, "channel": channel,
                                                  "address": address or None, "notes": notes or None}) is not None:
                st.rerun()
    if cs:
        st.markdown("**Ubah / hapus**")
        cmap = {f"{c['name']} (#{c['id']})": c for c in cs}
        c = cmap[st.selectbox("Pilih customer", list(cmap), key="ec_pick")]
        a, b = st.columns(2)
        name = a.text_input("Nama", value=c["name"], key=f"ec_name_{c['id']}")
        phone = b.text_input("Telepon/WA", value=c["phone"] or "", key=f"ec_phone_{c['id']}")
        address = a.text_input("Alamat", value=c["address"] or "", key=f"ec_addr_{c['id']}")
        notes = b.text_input("Catatan", value=c["notes"] or "", key=f"ec_notes_{c['id']}")
        u, d = st.columns(2)
        if u.button("Simpan perubahan", key="btn_edit_customer"):
            if call("PUT", f"/customers/{c['id']}", json={"name": name, "phone": phone or None, "address": address or None,
                                                          "notes": notes or None}) is not None:
                st.rerun()
        if d.button("Hapus customer", key="btn_del_customer"):
            if call("DELETE", f"/customers/{c['id']}") is not None:
                st.rerun()


def page_produk():
    st.header("Produk")
    ps = get("/products", active_only="false") or []
    if ps:
        st.dataframe(pd.DataFrame([{"SKU": p["sku"], "Nama": p["name"], "Kategori": p["category"],
                                    "Bisa custom": "ya" if p["is_customizable"] else "tidak",
                                    "Harga (Rp)": p["price_idr"], "Aktif": "ya" if p["active"] else "tidak"} for p in ps]),
                     hide_index=True)
        st.caption("Harga masih kosong sampai diisi sesuai katalog Firaza yang sebenarnya.")
    with st.form("form_add_product", clear_on_submit=True):
        st.markdown("**Tambah produk**")
        a, b = st.columns(2)
        sku, name = a.text_input("SKU", key="ap_sku"), b.text_input("Nama", key="ap_name")
        cat = a.selectbox("Kategori", ["sajadah", "handuk", "hampers", "packaging", "lainnya"], key="ap_cat")
        price = b.number_input("Harga (Rp, 0 = belum diisi)", min_value=0, step=1000, key="ap_price")
        custom = st.checkbox("Bisa dipersonalisasi", key="ap_custom")
        if st.form_submit_button("Simpan"):
            if not sku.strip() or not name.strip():
                st.error("SKU dan nama wajib diisi.")
            elif call("POST", "/products", json={"sku": sku.strip(), "name": name.strip(), "category": cat,
                                                 "is_customizable": custom, "price_idr": int(price) or None}) is not None:
                st.rerun()
    if ps:
        st.markdown("**Aktif / nonaktif**")
        pmap = {f"{p['sku']} · {p['name']}": p for p in ps}
        p = pmap[st.selectbox("Pilih produk", list(pmap), key="tp_pick")]
        if st.button("Nonaktifkan" if p["active"] else "Aktifkan", key="btn_toggle_product"):
            if call("PUT", f"/products/{p['id']}", json={"active": not p["active"]}) is not None:
                st.rerun()


# ================================================================== main
st.set_page_config(page_title="Firaza Custom Order", page_icon="🧵", layout="wide")
PAGES = {"Dashboard": page_dashboard, "Order Baru": page_order_baru, "Daftar Order": page_daftar_order,
         "Customer": page_customer, "Produk": page_produk}
with st.sidebar:
    st.title("🧵 Firaza")
    st.caption("Custom Order Management")
    choice = st.radio("Menu", list(PAGES), key="menu", label_visibility="collapsed")
    try:
        h = requests.get(API + "/health", timeout=10).json()
        st.divider()
        st.caption(f"Model klasifikasi: {'siap ✅' if h['classifier_loaded'] else 'belum dimuat ⚠️'}")
        st.caption(f"Gemini: {'aktif ✅' if h['gemini_configured'] else 'tidak aktif (memakai fallback)'}")
    except requests.exceptions.RequestException:
        st.error(f"API tidak dapat dihubungi di {API}")
        st.stop()
PAGES[choice]()