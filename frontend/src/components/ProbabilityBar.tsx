interface Props {
  probability: number;
}

function ProbabilityBar({ probability }: Props) {
  const percent = (probability * 100).toFixed(1);

  return (
    <div className="space-y-2">

      <div className="flex justify-between">
        <span>Default Probability</span>
        <span>{percent}%</span>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-4">

        <div
          className="bg-indigo-600 h-4 rounded-full transition-all duration-500"
          style={{
            width: `${percent}%`,
          }}
        ></div>

      </div>

    </div>
  );
}

export default ProbabilityBar;