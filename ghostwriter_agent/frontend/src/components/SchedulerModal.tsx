import React, { useState, useMemo } from 'react';
import { BG_MEDIUM } from '../constants/theme';
import { getDaysInMonth, getFirstDayOfMonth, TIME_SLOTS, TimeSlot } from '../utils/calendar';

interface SchedulerModalProps {
  title: string;
  onClose: () => void;
  onSchedule: (platform: string, content: string, date: string, timeKey: string, imageUrl: string | null) => void;
  currentContent: string;
  generatedImageUrl: string | null;
}

const SchedulerModal: React.FC<SchedulerModalProps> = ({ 
  title, 
  onClose, 
  onSchedule, 
  currentContent, 
  generatedImageUrl 
}) => {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedDate, setSelectedDate] = useState(today);
  const [selectedTimeKey, setSelectedTimeKey] = useState<string>(TIME_SLOTS[0].key);

  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();
  const daysInMonth = getDaysInMonth(year, month);
  const firstDay = getFirstDayOfMonth(year, month);
  const monthName = currentDate.toLocaleString('default', { month: 'long' });

  const calendarDays = useMemo(() => {
    const days = [];
    for (let i = 0; i < firstDay; i++) {
      days.push({ day: null, isCurrentMonth: false });
    }
    for (let day = 1; day <= daysInMonth; day++) {
      const date = new Date(year, month, day);
      date.setHours(0, 0, 0, 0);
      const isToday = date.getTime() === today.getTime();
      const isSelected = selectedDate && date.getTime() === selectedDate.getTime();
      const isPast = date < today;
      days.push({ day, date, isCurrentMonth: true, isToday, isSelected, isPast });
    }
    return days;
  }, [year, month, daysInMonth, firstDay, selectedDate, today]);

  const handlePrevMonth = () => {
    setCurrentDate(new Date(year, month - 1, 1));
  };

  const handleNextMonth = () => {
    setCurrentDate(new Date(year, month + 1, 1));
  };

  const handleDayClick = (dayItem: any) => {
    if (dayItem.day !== null && !dayItem.isPast) {
      setSelectedDate(dayItem.date);
    }
  };

  const handleTimeSelect = (timeKey: string) => {
    setSelectedTimeKey(timeKey);
  };

  const handleScheduleClick = () => {
    if (selectedDate && selectedTimeKey) {
      const yyyy = selectedDate.getFullYear();
      const mm = String(selectedDate.getMonth() + 1).padStart(2, '0');
      const dd = String(selectedDate.getDate()).padStart(2, '0');
      const dateString = `${yyyy}-${mm}-${dd}`;
      onSchedule(title, currentContent, dateString, selectedTimeKey, generatedImageUrl);
    }
  };

  const DayComponent: React.FC<{ item: any }> = ({ item }) => {
    if (item.day === null) {
      return <div className="p-2 aspect-square"></div>;
    }

    let classes = "p-2 aspect-square rounded-lg transition duration-150 flex items-center justify-center font-semibold text-sm cursor-pointer";

    if (item.isPast) {
      classes += " text-slate-600 line-through cursor-not-allowed";
    } else if (item.isSelected) {
      classes += " bg-emerald-600 text-white shadow-lg";
    } else if (item.isToday) {
      classes += " border border-emerald-500 text-emerald-400 hover:bg-slate-700";
    } else {
      classes += " text-white hover:bg-slate-700";
    }

    return (
      <div className={classes} onClick={() => handleDayClick(item)}>
        {item.day}
      </div>
    );
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-70 z-50 flex justify-center items-center backdrop-blur-sm">
      <div className={`w-full max-w-4xl p-8 rounded-2xl ${BG_MEDIUM} shadow-2xl border border-emerald-600/50`}>
        <h3 className="text-2xl font-bold text-white mb-6">Schedule {title} Post</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Date Picker */}
          <div className="bg-slate-700 p-4 rounded-xl shadow-inner">
            <div className="flex justify-between items-center mb-4">
              <button 
                onClick={handlePrevMonth}
                className="text-white hover:text-emerald-400 disabled:text-slate-600"
                disabled={year === today.getFullYear() && month === today.getMonth()}
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
                </svg>
              </button>
              <h4 className="text-xl font-bold text-emerald-400">{monthName} {year}</h4>
              <button onClick={handleNextMonth} className="text-white hover:text-emerald-400">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                </svg>
              </button>
            </div>

            <div className="grid grid-cols-7 gap-1 text-center text-sm font-medium text-slate-400 mb-2">
              {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                <div key={day} className="p-2">{day}</div>
              ))}
            </div>

            <div className="grid grid-cols-7 gap-1">
              {calendarDays.map((item, index) => (
                <DayComponent key={index} item={item} />
              ))}
            </div>
            <p className="mt-4 text-center text-sm text-slate-400">
              Selected Date: <span className="text-emerald-400 font-mono">
                {selectedDate.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })}
              </span>
            </p>
          </div>

          {/* Time Picker */}
          <div className="bg-slate-700 p-4 rounded-xl shadow-inner flex flex-col">
            <h4 className="text-xl font-bold text-white mb-4">Select Time (EST)</h4>
            <div className="flex-grow grid grid-cols-3 gap-2 overflow-y-auto max-h-[300px] p-1">
              {TIME_SLOTS.map(slot => (
                <button
                  key={slot.key}
                  onClick={() => handleTimeSelect(slot.key)}
                  className={`p-2 rounded-lg text-sm font-medium transition duration-150 
                    ${selectedTimeKey === slot.key 
                      ? 'bg-blue-600 text-white shadow-lg' 
                      : 'bg-slate-800 text-slate-300 hover:bg-slate-600'
                    }`}
                >
                  {slot.display}
                </button>
              ))}
            </div>
            <p className="mt-4 text-center text-sm text-slate-400">
              Selected Time: <span className="text-blue-400 font-mono">
                {TIME_SLOTS.find(s => s.key === selectedTimeKey)?.display}
              </span>
            </p>
          </div>
        </div>

        <div className="mt-8 flex justify-end space-x-4">
          <button
            onClick={onClose}
            className="px-6 py-3 bg-slate-700 text-white font-semibold rounded-xl hover:bg-slate-600 transition"
          >
            Cancel
          </button>
          <button
            onClick={handleScheduleClick}
            disabled={!selectedDate || !selectedTimeKey}
            className="px-6 py-3 bg-emerald-600 text-white font-bold rounded-xl hover:bg-emerald-700 transition disabled:bg-slate-700 disabled:text-slate-500"
          >
            Confirm Schedule
          </button>
        </div>
      </div>
    </div>
  );
};

export default SchedulerModal;

