interface Props {
  risk: string;
}

function RiskBadge({ risk }: Props) {
  let color = "bg-green-500";

  if (risk.toLowerCase().includes("medium")) {
    color = "bg-yellow-500";
  }

  if (risk.toLowerCase().includes("default") || risk.toLowerCase().includes("high")) {
    color = "bg-red-600";
  }

  return (
    <span
      className={`${color} text-white px-4 py-2 rounded-full font-semibold`}
    >
      {risk}
    </span>
  );
}

export default RiskBadge;