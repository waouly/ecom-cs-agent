"""状态码与中文标签。"""

ORDER_STATUS_LABEL = {
    "pending_payment": "待付款",
    "paid": "已付款待发货",
    "shipped": "已发货",
    "signed": "已签收",
    "completed": "已完成",
    "refunding": "退款中",
    "refunded": "已退款",
    "closed": "已关闭",
}

LOGISTICS_STATUS_LABEL = {
    "created": "已揽收",
    "in_transit": "运输中",
    "out_for_delivery": "派送中",
    "signed": "已签收",
    "exception": "配送异常",
}

AFTERSALE_STATUS_LABEL = {
    "pending_review": "待审核",
    "approved": "已通过",
    "rejected": "已驳回",
    "done": "已完成",
}

AFTERSALE_TYPE_LABEL = {
    "refund_only": "仅退款",
    "return_refund": "退货退款",
    "exchange": "换货",
    "repair": "维修",
}

CATEGORY_LABEL = {
    "apparel": "服装鞋帽",
    "digital": "数码电子",
    "custom": "定制商品",
    "underwear": "内衣贴身",
    "fresh": "生鲜食品",
    "furniture": "家具家装",
    "beauty": "美妆个护",
}
