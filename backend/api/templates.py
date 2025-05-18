from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/templates", tags=["templates"])

class Template(BaseModel):
    id: str
    name: str
    description: str
    imageUrl: Optional[str] = None
    isPremium: bool = False

# Sample templates data (replace with database in production)
templates = [
    Template(
        id="1",
        name="Standard Document",
        description="A clean, professional document template",
        imageUrl="/templates/standard.png",
        isPremium=False
    ),
    Template(
        id="2",
        name="Premium Report",
        description="Advanced report template with charts and graphs",
        imageUrl="/templates/premium.png",
        isPremium=True
    ),
]

@router.get("/", response_model=List[Template])
async def get_templates():
    return templates

@router.get("/{template_id}", response_model=Template)
async def get_template(template_id: str):
    template = next((t for t in templates if t.id == template_id), None)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return template
