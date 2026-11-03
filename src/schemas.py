"""
schemas.py - Skema Pydantic (v2) untuk API Firaza Custom Order Management.

Catatan kompatibilitas: venv pengembangan memakai Python 3.9, jadi memakai Optional[...] / List[...]
(bukan `X | None`) agar Pydantic bisa mengevaluasi anotasi saat runtime.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal

from database import CHANNELS, FONT_OPTIONS, LAYOUT_OPTIONS, ORDER_TYPES

OrderTypeLit = Literal["Ready Stock", "Custom Sajadah", "Custom Handuk", "Hampers", "Other"]
StatusLit = Literal["New", "Design/Preparation", "Production", "Completed"]


def _match_choice(value, choices, field):
    """Cocokkan tanpa membedakan huruf besar/kecil; kembalikan penulisan baku."""
    if value is None or value == "":
        return None
    for c in choices:
        if value.strip().lower() == c.lower():
            return c
    raise ValueError(f"{field} harus salah satu dari: {', '.join(choices)}")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ------------------------------------------------------------------ klasifikasi
class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["Mau sajadah navy 20 pcs untuk nikahan, custom nama Budi & Sarah"])


class ClassifyResponse(BaseModel):
    label: OrderTypeLit
    confidence: float
    needs_review: bool
    probabilities: Dict[str, float]


# ------------------------------------------------------------------ customer
class CustomerCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    phone: Optional[str] = Field(None, max_length=30)
    channel: str = "WhatsApp"
    address: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("channel")
    @classmethod
    def _channel(cls, v):
        return _match_choice(v, CHANNELS, "channel")


class CustomerUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    phone: Optional[str] = Field(None, max_length=30)
    channel: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("channel")
    @classmethod
    def _channel(cls, v):
        return _match_choice(v, CHANNELS, "channel")


class CustomerOut(ORMModel):
    id: int
    name: str
    phone: Optional[str]
    channel: str
    address: Optional[str]
    notes: Optional[str]
    created_at: datetime


# ------------------------------------------------------------------ produk
class ProductCreate(BaseModel):
    sku: str = Field(..., min_length=1, max_length=40)
    name: str = Field(..., min_length=1, max_length=120)
    category: Literal["sajadah", "handuk", "hampers", "packaging", "lainnya"]
    is_customizable: bool = False
    price_idr: Optional[int] = Field(None, ge=0)


class ProductUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    category: Optional[Literal["sajadah", "handuk", "hampers", "packaging", "lainnya"]] = None
    is_customizable: Optional[bool] = None
    price_idr: Optional[int] = Field(None, ge=0)
    active: Optional[bool] = None


class ProductOut(ORMModel):
    id: int
    sku: str
    name: str
    category: str
    is_customizable: bool
    price_idr: Optional[int]
    active: bool


# ------------------------------------------------------------------ item order
class OrderItemIn(BaseModel):
    product_id: Optional[int] = None
    quantity: int = Field(1, ge=1, le=100000)
    color: Optional[str] = Field(None, max_length=40)
    custom_text: Optional[str] = Field(None, max_length=200, description="Nama/teks yang dicetak atau dibordir")
    font: Optional[str] = None
    layout: Optional[str] = None
    vendor_id: Optional[int] = None
    item_notes: Optional[str] = None

    @field_validator("font")
    @classmethod
    def _font(cls, v):
        return _match_choice(v, FONT_OPTIONS, "font")

    @field_validator("layout")
    @classmethod
    def _layout(cls, v):
        return _match_choice(v, LAYOUT_OPTIONS, "layout")


class OrderItemUpdate(BaseModel):
    product_id: Optional[int] = None
    quantity: Optional[int] = Field(None, ge=1, le=100000)
    color: Optional[str] = Field(None, max_length=40)
    custom_text: Optional[str] = Field(None, max_length=200)
    font: Optional[str] = None
    layout: Optional[str] = None
    vendor_id: Optional[int] = None
    item_notes: Optional[str] = None

    @field_validator("font")
    @classmethod
    def _font(cls, v):
        return _match_choice(v, FONT_OPTIONS, "font")

    @field_validator("layout")
    @classmethod
    def _layout(cls, v):
        return _match_choice(v, LAYOUT_OPTIONS, "layout")


class OrderItemOut(ORMModel):
    id: int
    product_id: Optional[int]
    quantity: int
    color: Optional[str]
    custom_text: Optional[str]
    font: Optional[str]
    layout: Optional[str]
    vendor_id: Optional[int]
    item_notes: Optional[str]


# ------------------------------------------------------------------ order
class OrderCreate(BaseModel):
    customer_id: int
    raw_message: Optional[str] = Field(None, max_length=4000)
    channel: str = "WhatsApp"
    order_type: Optional[OrderTypeLit] = Field(None, description="Jenis final (konfirmasi admin); kosongkan bila belum")
    auto_classify: bool = Field(True, description="Klasifikasikan raw_message dengan model sendiri")
    confirm_prediction: bool = Field(False, description="Jika True dan order_type kosong, pakai hasil prediksi sebagai jenis final")
    due_date: Optional[datetime] = None
    notes: Optional[str] = None
    items: List[OrderItemIn] = []

    @field_validator("channel")
    @classmethod
    def _channel(cls, v):
        return _match_choice(v, CHANNELS, "channel")


class OrderUpdate(BaseModel):
    order_type: Optional[OrderTypeLit] = None
    channel: Optional[str] = None
    raw_message: Optional[str] = Field(None, max_length=4000)
    due_date: Optional[datetime] = None
    notes: Optional[str] = None

    @field_validator("channel")
    @classmethod
    def _channel(cls, v):
        return _match_choice(v, CHANNELS, "channel")


class StatusChange(BaseModel):
    status: StatusLit
    note: Optional[str] = None


class OrderOut(ORMModel):
    id: int
    order_code: str
    customer_id: int
    channel: str
    raw_message: Optional[str]
    predicted_type: Optional[OrderTypeLit]
    predicted_confidence: Optional[float]
    order_type: Optional[OrderTypeLit]
    status: StatusLit
    ai_summary: Optional[Dict[str, Any]]
    greeting_card: Optional[str]
    due_date: Optional[datetime]
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime
    items: List[OrderItemOut] = []

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v):
        return v.value if isinstance(v, Enum) else v


class StatusHistoryOut(ORMModel):
    id: int
    from_status: Optional[str]
    to_status: str
    changed_at: datetime
    note: Optional[str]


# ------------------------------------------------------------------ Generative AI
class SummaryItem(BaseModel):
    produk: str = Field(..., description="Nama produk (mis. sajadah, handuk, hampers)")
    jumlah: Optional[int] = Field(None, description="Jumlah pcs jika disebut customer")
    warna: Optional[str] = None
    teks_custom: Optional[str] = Field(None, description="Nama/teks yang diminta, persis seperti ditulis customer")
    font: Optional[str] = None
    layout: Optional[str] = None


class GeminiSummary(BaseModel):
    """Skema yang dikirim ke Gemini (structured output). Jenis order TIDAK dibuat Gemini: berasal dari model sendiri."""
    ringkasan: str = Field(..., description="Ringkasan 1-2 kalimat dalam Bahasa Indonesia")
    items: List[SummaryItem] = []
    acara: Optional[str] = None
    tenggat: Optional[str] = Field(None, description="Tenggat/tanggal jika disebut customer, apa adanya")
    perlu_klarifikasi: List[str] = Field([], description="Informasi penting yang belum jelas dan perlu ditanyakan admin")


class GreetingRequest(BaseModel):
    tone: Literal["hangat", "formal", "islami"] = "hangat"


class GreetingOut(BaseModel):
    kartu_ucapan: str
    sumber: Literal["gemini", "fallback"]
    peringatan: List[str] = []
