"""HTTP API: диалог, повестка, поиск, голос (распознавание и синтез на сервере), экспорт."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.api.schemas import AgendaResponse, ChatRequest, ChatResponse, HealthResponse, ItemOut, SpeechOut, TTSRequest
from app.container import Container
from app.core.models import Occurrence, Period
from app.services.ics import to_ics

router = APIRouter(prefix="/api")

USER_ID = Query("demo", min_length=1, max_length=50, pattern=r"^[\w.-]+$")
TIMEZONE = Query(None, max_length=64)


def get_container(request: Request) -> Container:
    return request.app.state.container


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, c: Container = Depends(get_container)) -> ChatResponse:
    now = c.now(body.timezone)
    reply = c.assistant.handle(body.user_id, body.session_id, body.text, now)
    return ChatResponse(
        text=reply.text,
        summary=reply.summary,
        speech=reply.speech,
        items=[ItemOut.from_occurrence(o, now.date()) for o in reply.items[:50]],
        numbered=reply.numbered,
        intent=reply.intent,
        awaiting=reply.awaiting,
        nlu=reply.nlu,
    )


@router.get("/agenda", response_model=AgendaResponse)
def agenda(
    user_id: str = USER_ID,
    start: date | None = None,
    end: date | None = None,
    tz: str | None = TIMEZONE,
    c: Container = Depends(get_container),
) -> AgendaResponse:
    today = c.now(tz).date()
    start = start or today
    end = end or start + timedelta(days=6)
    if end < start or (end - start).days > 366:
        raise HTTPException(422, "Интервал должен быть от 0 до 366 дней")
    occurrences = c.planner.agenda(user_id, Period(start, end))
    undated = [Occurrence(r, None) for r in c.planner.undated(user_id)]
    return AgendaResponse(start=start, end=end, items=[ItemOut.from_occurrence(o, today) for o in occurrences + undated])


@router.get("/search", response_model=list[ItemOut])
def search(
    q: str = Query(..., min_length=1, max_length=200),
    user_id: str = USER_ID,
    tz: str | None = TIMEZONE,
    c: Container = Depends(get_container),
) -> list[ItemOut]:
    now = c.now(tz)
    hits = c.planner.search(user_id, q, now=now, limit=20)
    return [ItemOut.from_occurrence(Occurrence(h.record, h.record.day), now.date()) for h in hits]


@router.get("/reminders/due", response_model=list[ItemOut])
def due_reminders(
    since: datetime,
    user_id: str = USER_ID,
    tz: str | None = TIMEZONE,
    c: Container = Depends(get_container),
) -> list[ItemOut]:
    """Напоминания, время которых наступило после `since` — браузер опрашивает раз в полминуты."""
    now = c.now(tz)
    since = max(since.replace(tzinfo=None), now - timedelta(hours=1))
    return [ItemOut.from_occurrence(o, now.date()) for o in c.planner.due_reminders(user_id, since, now)]


@router.get("/calendar.ics")
def calendar(user_id: str = USER_ID, c: Container = Depends(get_container)) -> Response:
    body = to_ics(c.records.all(user_id), c.settings.timezone)
    return Response(body, media_type="text/calendar; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{user_id}.ics"'})


MAX_AUDIO_BYTES = 2 * 1024 * 1024  # 25 секунд WAV 16 кГц моно — около 800 КБ


@router.post("/stt", response_model=SpeechOut)
async def speech_to_text(request: Request, c: Container = Depends(get_container)) -> SpeechOut:
    """Распознаёт речь: тело запроса — WAV (PCM 16 бит, моно), который записывает браузер."""
    if c.stt is None:
        raise HTTPException(503, "Распознавание речи на сервере выключено")
    body = await request.body()
    if not body:
        raise HTTPException(422, "Пустая запись")
    if len(body) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Запись слишком длинная")
    from app.voice.audio import AudioError

    try:
        text = await run_in_threadpool(c.stt.transcribe, body)
    except AudioError as exc:
        raise HTTPException(422, f"Не удалось прочитать запись: {exc}") from exc
    return SpeechOut(text=text)


@router.post("/tts")
async def text_to_speech(body: TTSRequest, c: Container = Depends(get_container)) -> Response:
    """Озвучивает ответ ассистента и возвращает WAV."""
    if c.tts is None:
        raise HTTPException(503, "Синтез речи на сервере выключен")
    audio = await run_in_threadpool(c.tts.synthesize, body.text)
    return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.get("/health", response_model=HealthResponse)
def health(user_id: str = USER_ID, c: Container = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        nlu=c.nlu.mode,
        models=c.models,
        server_stt=c.stt is not None,
        server_tts=c.tts is not None,
        timezone=c.settings.timezone,
        records=c.records.count(user_id),
    )
