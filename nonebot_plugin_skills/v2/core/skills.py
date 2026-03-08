import inspect
import importlib
import pkgutil
import sys
from typing import Callable, Dict, Any, List, Optional
from pydantic import BaseModel
from pathlib import Path
from google.genai import types

class SkillDef:
    def __init__(self, name: str, description: str, func: Callable, params_schema: dict):
        self.name = name
        self.description = description
        self.func = func
        self.params_schema = params_schema

class SkillManager:
    def __init__(self):
        self.skills: Dict[str, SkillDef] = {}

    def register(self, name: str, description: str, params_schema: Optional[dict] = None):
        """Decorator to register a function as a skill."""
        def decorator(func: Callable):
            schema = params_schema or {}
            self.skills[name] = SkillDef(name, description, func, schema)
            return func
        return decorator

    def load_from_directory(self, package_path: str, package_name: str):
        """Dynamically load all Python modules in a directory so decorators trigger."""
        path = Path(package_path)
        if not path.exists():
            return
        
        # Add the directory to sys.path so importlib can find it
        if str(path.parent) not in sys.path:
            sys.path.insert(0, str(path.parent))

        for _, module_name, _ in pkgutil.iter_modules([str(path)]):
            full_module_name = f"{package_name}.{module_name}"
            importlib.import_module(full_module_name)

    def get_llm_tools(self) -> List[types.ToolDict]:
        """Generate Google Gemini function calling tool schema."""
        tools: List[types.ToolDict] = []
        for name, skill in self.skills.items():
            tool = {
                "function_declarations": [
                    {
                        "name": name,
                        "description": skill.description,
                        "parameters": {
                            "type": "OBJECT",
                            "properties": skill.params_schema.get("properties", {}),
                            "required": skill.params_schema.get("required", [])
                        }
                    }
                ]
            }
            tools.append(tool)
        return tools

    async def execute(self, name: str, context: Optional[Any] = None, **kwargs) -> Any:
        if name not in self.skills:
            return f"Skill {name} not found."
        
        skill = self.skills[name]
        try:
            # 注入 context 到参数中，如果函数支持的话
            sig = inspect.signature(skill.func)
            if "context" in sig.parameters:
                kwargs["context"] = context

            if inspect.iscoroutinefunction(skill.func):
                result = await skill.func(**kwargs)
            else:
                result = skill.func(**kwargs)
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

skill_manager = SkillManager()
