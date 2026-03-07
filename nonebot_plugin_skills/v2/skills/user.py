from typing import Optional
from nonebot import logger
from nonebot_plugin_skills.v2.core.skills import skill_manager
from nonebot_plugin_skills.v2.core.context import SkillContext

user_schema = {
    "properties": {
        "qq": {"type": "STRING", "description": "要查询的QQ号。如果要查询机器人(嘉然)自己的头像，必须传入机器人自己的QQ号。如果不传，则默认为当前发消息的用户。"}
    }
}

group_schema = {
    "properties": {
        "group_id": {"type": "STRING", "description": "要查询的群号。如果不传，默认为当前所在的群。"}
    }
}

@skill_manager.register("get_user_avatar", "获取指定QQ号的高清头像图片。可以用来展示用户或嘉然自己的头像。", user_schema)
async def get_user_avatar(qq: Optional[str] = None, context: Optional[SkillContext] = None) -> str:
    if not context: return "无法获取上下文。"
    
    target_qq = str(qq).strip() if qq else str(context.user_id)
    url = f"https://q.qlogo.cn/headimg_dl?dst_uin={target_qq}&spec=640"
    
    from nonebot_plugin_skills.v2.core.nlp import _get_image_data
    from nonebot_plugin_skills.v2.core.utils import save_image_to_local
    
    img_data = await _get_image_data(url)
    if not img_data: return "获取头像图片失败了呢..."
    
    return save_image_to_local(img_data)

@skill_manager.register("get_group_avatar", "获取QQ群的群头像图片", group_schema)
async def get_group_avatar(group_id: Optional[str] = None, context: Optional[SkillContext] = None) -> str:
    if not context: return "无法获取上下文。"
    
    target_group = str(group_id).strip() if group_id else ""
    if not target_group:
        target_group = str(getattr(context.event, "group_id", ""))
        
    if not target_group: return "嘉然没找到群号呢~"
        
    url = f"https://p.qlogo.cn/gh/{target_group}/{target_group}/640"
    
    from nonebot_plugin_skills.v2.core.nlp import _get_image_data
    from nonebot_plugin_skills.v2.core.utils import save_image_to_local
    
    img_data = await _get_image_data(url)
    if not img_data: return "获取群头像失败了..."
    
    return save_image_to_local(img_data)

@skill_manager.register("user_info", "查询QQ用户（群成员或好友）的详细个人信息，包括昵称、名片、性别、年龄、角色等。", user_schema)
async def user_info(qq: Optional[str] = None, context: Optional[SkillContext] = None) -> str:
    if not context or not context.bot: return "无法获取机器人连接。"
    
    target_qq = int(qq) if qq else int(context.user_id)
    group_id = getattr(context.event, "group_id", None)
    
    try:
        res = f"查询到 QQ {target_qq} 的详细信息如下：\n"
        
        # 1. 优先尝试获取群成员信息 (如果是在群里)
        if group_id:
            try:
                info = await context.bot.get_group_member_info(group_id=group_id, user_id=target_qq, no_cache=True)
                nickname = info.get("nickname", "未知")
                card = info.get("card", "")
                role = info.get("role", "member")
                sex = info.get("sex", "unknown")
                age = info.get("age", 0)
                join_time = info.get("join_time", 0)
                level = info.get("level", "")
                
                sex_str = "男" if sex == "male" else ("女" if sex == "female" else "未知")
                role_str = "群主" if role == "owner" else ("管理员" if role == "admin" else "普通成员")

                res += f"昵称: {nickname}\n"
                if card: res += f"群名片: {card}\n"
                res += f"性别: {sex_str}\n"
                res += f"年龄: {age}\n"
                res += f"群角色: {role_str}\n"
                if level: res += f"等级: {level}\n"
                return res
            except Exception:
                pass # 如果不是群成员，降级去查陌生人信息

        # 2. 尝试获取陌生人/好友信息
        info = await context.bot.get_stranger_info(user_id=target_qq)
        nickname = info.get("nickname", "未知")
        sex = info.get("sex", "unknown")
        age = info.get("age", 0)
        sex_str = "男" if sex == "male" else ("女" if sex == "female" else "未知")
        
        res += f"昵称: {nickname}\n性别: {sex_str}\n年龄: {age}"
        return res
        
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")
        return f"查询 QQ {target_qq} 的信息失败了呢: {e}"

@skill_manager.register("get_group_member_list", "获取当前群的所有成员列表概要，用于了解群内有哪些人。", group_schema)
async def get_group_member_list(group_id: Optional[str] = None, context: Optional[SkillContext] = None) -> str:
    if not context or not context.bot: return "无法获取机器人连接。"
    
    target_group = int(group_id) if group_id else getattr(context.event, "group_id", None)
    if not target_group: return "嘉然不知道要在哪个群查呢~"
    
    try:
        members = await context.bot.get_group_member_list(group_id=target_group)
        count = len(members)
        
        # 简单列出前 30 个成员防止消息过长
        sample = members[:30]
        member_names = [m.get("card") or m.get("nickname") or str(m.get("user_id")) for m in sample]
        
        res = f"该群共有 {count} 名成员。\n部分成员如下：\n" + "、".join(member_names)
        if count > 30:
            res += f"\n... 以及其他 {count - 30} 名成员。"
            
        return res
    except Exception as e:
        logger.error(f"Error fetching member list: {e}")
        return f"获取群成员列表失败: {e}"
