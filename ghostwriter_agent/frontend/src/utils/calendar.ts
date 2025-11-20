// Calendar utility functions

export const getDaysInMonth = (year: number, month: number): number => 
  new Date(year, month + 1, 0).getDate();

export const getFirstDayOfMonth = (year: number, month: number): number => 
  new Date(year, month, 1).getDay(); // 0=Sun, 6=Sat

export const formatTime = (hour: number, minute: number): string => {
  const period = hour >= 12 ? 'PM' : 'AM';
  const h = hour % 12 === 0 ? 12 : hour % 12;
  const m = minute.toString().padStart(2, '0');
  return `${h}:${m} ${period}`;
};

export interface TimeSlot {
  key: string;
  display: string;
}

export const generateTimeSlots = (): TimeSlot[] => {
  const slots: TimeSlot[] = [];
  for (let h = 8; h <= 18; h++) { // From 8 AM to 6 PM (inclusive)
    slots.push({ key: `${h}:00`, display: formatTime(h, 0) });
    if (h < 18) { // Only add 30 past for slots before 6 PM
      slots.push({ key: `${h}:30`, display: formatTime(h, 30) });
    }
  }
  return slots;
};

export const TIME_SLOTS = generateTimeSlots();

