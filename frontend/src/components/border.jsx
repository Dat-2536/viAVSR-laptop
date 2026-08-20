

const ItemBorder = ({ children, className = '' }) => {
  return (
    <div className={`flex-1 mx-[5px] my-[3px] h-[800px] border border-gray-400 rounded-[5px] ${className}`}>
      {children}
    </div>
  );
};


export default ItemBorder;



