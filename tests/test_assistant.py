from datetime import date, time

from app.core.models import Period, Record, Recurrence, RecordType, Status
from app.services.assistant import Assistant
from app.services.phrases import found_word
from app.storage.repository import DialogStateRepository
from tests.conftest import FRIDAY, NOW, Dialog


def records(dialog):
    return dialog.container.records.all(dialog.user)


def test_create_task_reminder_meeting(clean_dialog):
    r1 = clean_dialog.say("Напомни мне завтра в 10 утра позвонить Ивану")
    r2 = clean_dialog.say("В пятницу нужно отправить отчет")
    r3 = clean_dialog.say("Добавь встречу с командой завтра в 15:00")
    assert "напомню завтра, 14 октября, в 10:00" in r1.text
    assert "Отправить отчет" in r2.text and "пятницу, 16 октября" in r2.text
    assert "Встреча с командой" in r3.text
    saved = {(r.type, r.title, r.day, r.at) for r in records(clean_dialog)}
    assert saved == {
        (RecordType.REMINDER, "Позвонить Ивану", date(2026, 10, 14), time(10, 0)),
        (RecordType.TASK, "Отправить отчет", date(2026, 10, 16), None),
        (RecordType.MEETING, "Встреча с командой", date(2026, 10, 14), time(15, 0)),
    }


def test_clarifies_missing_time(clean_dialog):
    reply = clean_dialog.say("Поставь встречу с Иваном завтра")
    assert reply.text == "На какое время поставить встречу?" and reply.awaiting == "slot"
    reply = clean_dialog.say("в 15:00")
    assert reply.awaiting is None
    (record,) = records(clean_dialog)
    assert (record.title, record.day, record.at) == ("Встреча с Иваном", date(2026, 10, 14), time(15, 0))


def test_clarifies_missing_title_then_time(clean_dialog):
    assert clean_dialog.say("Напомни мне").text == "О чём напомнить?"
    assert clean_dialog.say("позвонить маме").text == "Когда напомнить?"
    clean_dialog.say("через 2 часа")
    (record,) = records(clean_dialog)
    assert (record.title, record.day, record.at) == ("Позвонить маме", NOW.date(), time(14, 0))


def test_new_command_cancels_pending_question(clean_dialog):
    clean_dialog.say("Поставь встречу с Иваном завтра")
    reply = clean_dialog.say("Что у меня сегодня?")
    assert reply.intent == "list_events" and reply.awaiting is None
    assert records(clean_dialog) == []


def test_meeting_conflict_offers_choice(clean_dialog):
    clean_dialog.say("Добавь встречу с командой завтра в 15:00")
    reply = clean_dialog.say("Поставь встречу с Иваном завтра в 15:30")
    assert "уже запланировано" in reply.text and reply.awaiting == "confirm"
    assert clean_dialog.say("нет").awaiting == "slot"
    clean_dialog.say("в 17")
    assert {r.at for r in records(clean_dialog)} == {time(15, 0), time(17, 0)}


def test_new_command_with_own_title_is_not_an_answer(clean_dialog):
    clean_dialog.say("Добавь встречу с командой завтра в 15:00")
    assert clean_dialog.say("Добавь встречу с командой завтра в 15:30").awaiting == "confirm"
    assert clean_dialog.say("нет").text == "Хорошо. На какое время тогда поставить?"
    reply = clean_dialog.say("Каждый понедельник в 10 планёрка")
    assert "Планёрка" in reply.text and reply.awaiting is None
    saved = {(r.title, r.day, r.at, r.recurrence) for r in records(clean_dialog)}
    assert saved == {
        ("Встреча с командой", date(2026, 10, 14), time(15, 0), None),
        ("Планёрка", date(2026, 10, 19), time(10, 0), Recurrence.WEEKLY),
    }


def test_create_verb_without_title_is_still_an_answer(clean_dialog):
    assert clean_dialog.say("Поставь встречу с Иваном завтра").awaiting == "slot"
    clean_dialog.say("поставь на 15:00")
    (record,) = records(clean_dialog)
    assert (record.title, record.day, record.at) == ("Встреча с Иваном", date(2026, 10, 14), time(15, 0))


def test_reminder_time_answer(clean_dialog):
    assert clean_dialog.say("напомни купить молоко").text == "Когда напомнить?"
    clean_dialog.say("в 8 вечера")
    (record,) = records(clean_dialog)
    assert (record.type, record.title, record.at) == (RecordType.REMINDER, "Купить молоко", time(20, 0))


def test_unclear_phrase_with_date_is_not_a_task(empty_container):
    dialog = Dialog(empty_container, user="tester", now=FRIDAY)
    reply = dialog.say("чё по планам на завтра")
    assert reply.intent == "list_events" and records(dialog) == []


def test_implicit_create_of_non_action_asks_first(empty_container):
    dialog = Dialog(empty_container, user="tester", now=FRIDAY)
    reply = dialog.say("завтра Петров")
    assert reply.text.startswith("Создать задачу «Петров» на завтра") and reply.awaiting == "confirm"
    assert records(dialog) == []
    assert dialog.say("нет").text == "Хорошо, не создаю. Что сделать?"
    assert records(dialog) == []
    dialog.say("завтра Петров")
    assert "Петров" in dialog.say("да").text
    (record,) = records(dialog)
    assert (record.title, record.day) == ("Петров", date(2026, 10, 24))


def test_implicit_create_of_action_is_silent(empty_container):
    dialog = Dialog(empty_container, user="tester", now=FRIDAY)
    assert dialog.say("мне надо в субботу забрать посылку").awaiting is None
    assert dialog.say("в среду отчёт").awaiting is None
    assert {(r.title, r.day) for r in records(dialog)} == {
        ("Забрать посылку", date(2026, 10, 24)), ("Отчёт", date(2026, 10, 28)),
    }


def test_past_time_today_is_moved_to_tomorrow_on_confirm(clean_dialog):
    reply = clean_dialog.say("Напомни сегодня в 9 выпить витамины")
    assert "уже прошло" in reply.text
    clean_dialog.say("да")
    (record,) = records(clean_dialog)
    assert (record.day, record.at) == (date(2026, 10, 14), time(9, 0))


def test_dialog_context_move_it(clean_dialog):
    clean_dialog.say("Добавь встречу с командой завтра в 15:00")
    reply = clean_dialog.say("А перенеси её на пятницу")
    assert "пятницу, 16 октября" in reply.text
    (record,) = records(clean_dialog)
    assert (record.day, record.at) == (date(2026, 10, 16), time(15, 0))
    clean_dialog.say("сдвинь её на час позже")
    assert records(clean_dialog)[0].at == time(16, 0)


def test_list_period(dialog):
    reply = dialog.say("Что запланировано на эту неделю?")
    agenda = dialog.container.planner.agenda("demo", Period(date(2026, 10, 12), date(2026, 10, 18)))
    assert reply.intent == "list_events" and len(reply.items) == len(agenda) > 0


def test_list_empty_mentions_nothing(clean_dialog):
    assert "ничего не запланировано" in clean_dialog.say("Что у меня сегодня?").text


def test_search_and_when_question(dialog):
    reply = dialog.say("Найди задачу про презентацию")
    assert reply.intent == "search" and reply.items
    assert all("презентац" in o.record.title.lower() for o in reply.items)
    reply = dialog.say("Когда у меня была встреча с Иваном?")
    assert reply.text.startswith("Последний раз —")
    assert reply.items[0].record.type == RecordType.MEETING and reply.items[0].record.starts_at < NOW


def test_search_respects_record_type(dialog):
    reply = dialog.say("Найди задачу про презентацию")
    assert reply.items and {o.record.type for o in reply.items} == {RecordType.TASK}
    assert "по запросу «презентацию»" in reply.text and "«про " not in reply.text
    reply = dialog.say("Найди напоминание про договор")
    assert reply.items and {o.record.type for o in reply.items} == {RecordType.REMINDER}
    reply = dialog.say("найди всё про хакатон")
    assert len({o.record.type for o in reply.items}) > 1


def test_search_says_how_many_are_shown(dialog):
    reply = dialog.say("найди всё про хакатон")
    found = len(dialog.container.planner.search("demo", "хакатон", now=NOW, limit=None))
    assert found > len(reply.items) == 30
    assert reply.text.startswith(f"Нашёл {found_word(found)} по запросу «хакатон», показываю 30 ближайших")


def test_task_search_falls_back_to_reminders(clean_dialog):
    clean_dialog.say("Напомни завтра в 10 сдать отчёт")
    reply = clean_dialog.say("Найди задачу про отчёт")
    assert [o.record.type for o in reply.items] == [RecordType.REMINDER]
    assert reply.text.startswith("Нашёл 1 задачу по запросу «отчёт»")


def test_delete_with_choice_and_confirmation(dialog):
    before = len(records(dialog))
    reply = dialog.say("Удали задачу про отчёт")
    assert reply.awaiting == "choose" and reply.numbered
    chosen = reply.items[1].record
    assert dialog.say("вторую").awaiting == "confirm"
    assert "Удалил" in dialog.say("да").text
    assert len(records(dialog)) == before - 1
    assert dialog.container.records.get("demo", chosen.id) is None


def test_refers_to_record_deleted_from_agenda(clean_dialog):
    clean_dialog.say("Напомни завтра в 10 позвонить Ивану")
    (record,) = records(clean_dialog)
    clean_dialog.container.planner.delete(record)  # так удаляет кнопка в повестке
    assert clean_dialog.say("удали её").text == "Эта запись уже удалена."


def test_pending_delete_of_record_deleted_from_agenda(clean_dialog):
    clean_dialog.say("Напомни завтра в 10 позвонить Ивану")
    assert clean_dialog.say("удали её").awaiting == "confirm"
    clean_dialog.container.planner.delete(records(clean_dialog)[0])
    assert clean_dialog.say("да").text == "Эта запись уже удалена."


def maria_dialog(clean_dialog, *days):
    for day in days:
        clean_dialog.say(f"Поставь встречу с Марией {day} в 15:00")
    reply = clean_dialog.say("Удали встречу с Марией")
    assert reply.awaiting == "choose" and len(reply.items) == len(days)
    return reply


def test_yes_picks_the_only_record_in_next_week(clean_dialog):
    maria_dialog(clean_dialog, "в пятницу", "20 ноября")
    reply = clean_dialog.say("да")
    assert reply.text == "Удалить встречу «Встреча с Марией» на пятницу, 16 октября, в 15:00?"


def test_yes_to_choice_asks_for_number_without_repeating_list(clean_dialog):
    maria_dialog(clean_dialog, "в пятницу", "в субботу")
    reply = clean_dialog.say("да")
    assert reply.text == "Назовите номер: первую, вторую…" and reply.awaiting == "choose" and not reply.items
    assert clean_dialog.say("вторую").text.startswith("Удалить встречу «Встреча с Марией» на субботу")


def test_complete_task(clean_dialog):
    clean_dialog.say("Поставь задачу купить билеты до конца недели")
    reply = clean_dialog.say("Отметь задачу про билеты выполненной")
    assert "выполненное" in reply.text
    assert records(clean_dialog)[0].status == Status.DONE


def test_recurring_meeting_appears_every_week(clean_dialog):
    clean_dialog.say("каждый понедельник в 10 планёрка")
    (record,) = records(clean_dialog)
    assert record.recurrence == Recurrence.WEEKLY
    occurrences = clean_dialog.container.planner.agenda("tester", Period(date(2026, 10, 12), date(2026, 11, 8)))
    assert [o.day for o in occurrences] == [date(2026, 10, 19), date(2026, 10, 26), date(2026, 11, 2)]


def test_state_survives_restart(empty_container):
    first = Dialog(empty_container, user="tester")
    first.say("Поставь встречу с Иваном завтра")
    # Новый экземпляр ассистента поверх той же базы — как после перезапуска сервера.
    c = empty_container
    restarted = Assistant(c.nlu, c.planner, DialogStateRepository(c.db))
    reply = restarted.handle("tester", "s1", "в 16:00", NOW)
    assert "Встреча с Иваном" in reply.text and reply.awaiting is None


def test_users_are_isolated(empty_container):
    Dialog(empty_container, user="alice").say("Поставь задачу купить молоко")
    assert empty_container.records.all("bob") == []
    assert "ничего не запланировано" in Dialog(empty_container, user="bob").say("Что у меня на этой неделе?").text


def test_when_only_asks_what_to_do(clean_dialog):
    assert clean_dialog.say("завтра в 10").text == "Что нужно сделать?"
    clean_dialog.say("купить цветы")
    (record,) = records(clean_dialog)
    assert (record.title, record.day, record.at) == ("Купить цветы", date(2026, 10, 14), time(10, 0))


def test_small_talk(clean_dialog):
    assert clean_dialog.say("спасибо").text.startswith("Пожалуйста")
    assert "задачи" in clean_dialog.say("привет").text


def past_records_dialog(clean_dialog):
    dialog = Dialog(clean_dialog.container, user="tester", now=FRIDAY)
    planner = dialog.container.planner
    meeting = planner.create(Record(type=RecordType.MEETING, title="Встреча с командой", user_id="tester",
                                    day=date(2026, 10, 22), at=time(15, 0)), FRIDAY)
    reminder = planner.create(Record(type=RecordType.REMINDER, title="Оплатить связь", user_id="tester",
                                     day=date(2026, 10, 20), at=time(9, 0)), FRIDAY)
    return dialog, meeting, reminder


def test_delete_yesterday_meeting(clean_dialog):
    dialog, meeting, _ = past_records_dialog(clean_dialog)
    assert dialog.say("удали вчерашнюю встречу").text.startswith("Удалить встречу «Встреча с командой» на вчера")
    dialog.say("да")
    assert dialog.container.records.get("tester", meeting.id) is None


def test_delete_past_date_without_year_means_nearest(clean_dialog):
    # 20 октября при «сегодня» 23-е — прошедшее 20-е, а не 20 октября следующего года.
    dialog, _, reminder = past_records_dialog(clean_dialog)
    reply = dialog.say("удали напоминание за 20 октября")
    assert reply.text == "Удалить напоминание «Оплатить связь» на вторник, 20 октября, в 09:00?", reply.text
    dialog.say("да")
    assert dialog.container.records.get("tester", reminder.id) is None


def test_this_reminder_when_focus_is_meeting(clean_dialog):
    # Ближайшее — встреча (она в фокусе), но в списке есть и напоминание: «это напоминание» — оно.
    clean_dialog.say("Добавь встречу с командой завтра в 9:00")
    clean_dialog.say("Напомни завтра в 10 позвонить Ивану")
    reply = clean_dialog.say("что у меня будет?")
    assert reply.text.startswith("Ближайшее событие — завтра, 14 октября, в 09:00: «Встреча с командой»")
    reply = clean_dialog.say("удали это напоминание")
    assert reply.text == "Удалить напоминание «Позвонить Ивану» на завтра, 14 октября, в 10:00?", reply.text


def test_this_reminder_without_reminder_in_context_asks(clean_dialog):
    clean_dialog.say("Напомни завтра в 10 позвонить Ивану")
    clean_dialog.say("Добавь встречу с командой завтра в 15:00")
    reply = clean_dialog.say("удали это напоминание")
    assert reply.awaiting is None and reply.text.startswith("Уточните, какое напоминание удалить")
    assert len(records(clean_dialog)) == 2
