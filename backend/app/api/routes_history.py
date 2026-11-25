"""
Translation History API Routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, desc
from typing import Optional, List
from backend.app.database import get_db
from backend.app.models.db_models import TranslationHistory

router = APIRouter(prefix="/api/history", tags=["History"])

@router.get("")
async def get_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    module: Optional[str] = Query(None, description="Filter by module: 'text', 'voice', 'isl'"),
    search: Optional[str] = Query(None, description="Search term in source or translated text"),
    favorites_only: bool = Query(False),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves paginated translation history with optional filters."""
    query = select(TranslationHistory).order_by(desc(TranslationHistory.created_at))

    if module:
        query = query.where(TranslationHistory.module == module)
    if favorites_only:
        query = query.where(TranslationHistory.is_favorite == True)
    if search:
        search_fmt = f"%{search}%"
        query = query.where(
            (TranslationHistory.source_text.ilike(search_fmt)) |
            (TranslationHistory.translated_text.ilike(search_fmt))
        )

    query = query.offset(offset).limit(limit)
    res = await db.execute(query)
    records = res.scalars().all()

    return {
        "status": "success",
        "count": len(records),
        "history": [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "source_language": r.source_language,
                "target_language": r.target_language,
                "source_text": r.source_text,
                "translated_text": r.translated_text,
                "module": r.module,
                "confidence": r.confidence,
                "pipeline_notes": r.pipeline_notes,
                "is_favorite": r.is_favorite
            }
            for r in records
        ]
    }

@router.post("/{history_id}/favorite")
async def toggle_favorite(history_id: int, db: AsyncSession = Depends(get_db)):
    """Toggles favorite status for a translation record."""
    res = await db.execute(select(TranslationHistory).where(TranslationHistory.id == history_id))
    record = res.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="History record not found")

    record.is_favorite = not record.is_favorite
    await db.commit()
    await db.refresh(record)

    return {
        "status": "success",
        "id": record.id,
        "is_favorite": record.is_favorite
    }

@router.delete("/{history_id}")
async def delete_history_item(history_id: int, db: AsyncSession = Depends(get_db)):
    """Deletes a specific history record."""
    res = await db.execute(select(TranslationHistory).where(TranslationHistory.id == history_id))
    record = res.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail="History record not found")

    await db.delete(record)
    await db.commit()
    return {"status": "success", "message": f"Deleted record {history_id}"}

@router.delete("")
async def clear_all_history(db: AsyncSession = Depends(get_db)):
    """Clears all translation history."""
    await db.execute(delete(TranslationHistory))
    await db.commit()
    return {"status": "success", "message": "All history cleared"}

@router.get("/export")
async def export_history(db: AsyncSession = Depends(get_db)):
    """Exports entire history as structured JSON."""
    res = await db.execute(select(TranslationHistory).order_by(desc(TranslationHistory.created_at)))
    records = res.scalars().all()
    return {
        "exported_at": "now",
        "total": len(records),
        "data": [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "source_language": r.source_language,
                "target_language": r.target_language,
                "source_text": r.source_text,
                "translated_text": r.translated_text,
                "module": r.module,
                "confidence": r.confidence,
                "pipeline_notes": r.pipeline_notes,
                "is_favorite": r.is_favorite
            }
            for r in records
        ]
    }
