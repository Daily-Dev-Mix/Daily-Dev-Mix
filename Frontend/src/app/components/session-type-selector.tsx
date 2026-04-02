import { sessionTypes, SessionType } from '../utils/mockData';
import { motion } from 'motion/react';

interface SessionTypeSelectorProps {
  selectedType: SessionType | null;
  onSelectType: (type: SessionType) => void;
}

export function SessionTypeSelector({ selectedType, onSelectType }: SessionTypeSelectorProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      {sessionTypes.map((type, index) => (
        <motion.button
          key={type.value}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.05 }}
          onClick={() => onSelectType(type.value)}
          className={`
            relative p-6 rounded-lg transition-all text-left
            ${selectedType === type.value 
              ? 'bg-[#1DB954] shadow-lg shadow-[#1DB954]/20' 
              : 'bg-[#181818] hover:bg-[#282828]'
            }
          `}
        >
          <div className="text-4xl mb-3">{type.emoji}</div>
          <div className={`font-bold text-lg ${selectedType === type.value ? 'text-black' : 'text-white'}`}>
            {type.label}
          </div>
        </motion.button>
      ))}
    </div>
  );
}