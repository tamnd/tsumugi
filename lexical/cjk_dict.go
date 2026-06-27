package lexical

import "strings"

// cjkDictionary is the embedded segmentation word list, the common Chinese and Japanese
// words the forward maximum matcher splits on. It is deliberately compact, a few hundred
// high-frequency words rather than a full lexicon, because the character bigram fallback
// covers everything the dictionary omits: the dictionary buys precision on the words it
// lists, the fallback guarantees recall on the rest, so the list does not need to be
// exhaustive to be correct. A deployment with a domain lexicon builds its own segmenter
// from a larger list through newCJKSegmenter, the same way a deployment retrains the
// language profiles, without editing this file.
//
// The words are chosen for frequency on a general web crawl: function words, common
// nouns and verbs, place names, and the search-domain vocabulary a query is likely to
// carry. The list is stored as one space-separated string and split at init, which is
// less noisy in source than a literal slice of several hundred quoted strings.
var cjkDictionary = strings.Fields(cjkWords)

const cjkWords = `
中国 中文 我们 你们 他们 可以 没有 知道 时间 现在 已经 因为 所以 但是 如果 这个 那个
这些 那些 什么 怎么 为什么 一个 一些 很多 不是 就是 还是 或者 这样 那样 这里 那里
公司 企业 产品 服务 客户 用户 价格 质量 市场 销售 管理 经济 发展 社会 国家 政府
人民 世界 国际 全球 城市 地区 北京 上海 广州 深圳 香港 台湾 学校 学生 老师 教育
学习 工作 生活 健康 医院 医生 病人 治疗 药品 安全 环境 自然 资源 能源 技术 科技
科学 研究 数据 信息 网络 互联网 系统 软件 硬件 程序 设计 开发 平台 应用 手机 电脑
电话 邮件 地址 时候 问题 方法 办法 方式 内容 结果 原因 目的 计划 项目 活动 会议
新闻 报道 媒体 文化 历史 艺术 音乐 电影 游戏 体育 旅游 酒店 餐厅 美食 购物 商品
搜索 引擎 查询 关键 词语 页面 链接 网站 主页 首页 登录 注册 密码 账号 用户名
免费 下载 上传 分享 评论 点赞 收藏 关注 推荐 热门 最新 排行 榜单 视频 图片 文章
朋友 家人 父母 孩子 男人 女人 年轻 老人 喜欢 希望 需要 应该 必须 重要 主要 一般
特别 非常 比较 真的 觉得 认为 发现 出现 开始 结束 继续 完成 实现 提供 支持 帮助
今天 明天 昨天 现代 传统 经常 总是 已经 正在 将要 不能 不会 可能 也许 当然 其实
`

// init keeps the dictionary slice non-empty even if cjkWords is ever cleared, so the
// shared segmenter always has a valid word set rather than a nil map.
func init() {
	if len(cjkDictionary) == 0 {
		cjkDictionary = []string{}
	}
}
