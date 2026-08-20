import { AudioLines } from "lucide-react"


const Logo = () => {
  return <div className="w-[70px] h-[70px] rounded-[10px] bg-blue-500 flex items-center justify-center">
    <AudioLines className="w-[60px] w-[60px] text-white" />
  </div>
}




const MyHeader = () => {
  const name = "Vietnamese AVSR - Record & Infer Demo";
  const techUsed = "AV-HuBERT + CTC/Attention";
  return (<div className="flex p-[10px]">
    <Logo />
    <div className="flex flex-col ml-[10px] items-start">
      <h1 className="text-bold text-[25px]">{name}</h1>
      <p className="text-[20px]">{techUsed}</p>
    </div>
  </div>)
}



export default MyHeader;
