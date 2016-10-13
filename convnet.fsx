type T = float * float
type Vol = T [,,]

let emptyVol:Vol = Array3D.zeroCreate<T> 0 0 0

type ParamsGrads = {
  params : float [,,]
  grads: float [,,]
  l1DecayMul : float
  l2DecayMul : float
}


type InputCore = {
  outDepth : int
  outSx : int
  outSy : int
  inAct : Vol
  outAct : Vol  
} 

type ConvCore = {

  outDepth : int 
  sx : int
  inDepth : int
  inSx : int
  inSy : int
  sy : int
  stride : int
  pad : int
  l1DecayMul : float
  l2DecayMul : float
  outSx : int
  outSy : int
  filters : Vol list
  biases : Vol

  inAct : Vol
  outAct : Vol
} 

type FullyConnCore = {

  outDepth : int
  l1DecayMul : float
  l2DecayMul : float
  numInputs : int
  outSx : int
  outSy : int
  filters : Vol list
  biases : Vol
  inAct : Vol
  outAct : Vol
} 

type DropoutCore = {


  outSx : int
  outSy : int

  outDepth : int 
  dropProb : float
  dropped : float []
  inAct : Vol
  outAct : Vol
} 

type SoftmaxCore = {



  numInputs : int

  outSx : int
  outSy : int

  outDepth : int 


  dropProb : float
  dropped : float []
  inAct : Vol
  outAct : Vol
} 


type RegressionCore = {



  numInputs : int

  outSx : int
  outSy : int

  outDepth : int 


  dropProb : float
  dropped : float []
  inAct : Vol
  outAct : Vol
} 

type SVMCore = {



  numInputs : int

  outSx : int
  outSy : int

  outDepth : int 


  dropProb : float
  dropped : float []
  inAct : Vol
  outAct : Vol
} 

type ReluCore = {


  outSx : int
  outSy : int

  outDepth : int 
  inAct : Vol
  outAct : Vol
} 

type SigmoidCore = {


  outSx : int
  outSy : int

  outDepth : int 

  inAct : Vol
  outAct : Vol
} 

type MaxoutCore = {


  outSx : int
  outSy : int
  groupSize : int
  outDepth : int 

  switches : int []

  inAct : Vol
  outAct : Vol
} 

type TanhCore = {


  outSx : int
  outSy : int

  outDepth : int 
  inAct : Vol
  outAct : Vol
} 

type LocalResponseNormalizationCore = {



  k : int
  n : int
  alpha : float
  beta : float 

  outSx : int
  outSy : int
  outDepth : int 

  inAct : Vol
  outAct : Vol
} 

type PoolCore = {


  sx : int
  inDepth : int
  inSx : int
  inSy : int

  sy : int
  stride : int
  pad : int

  l1DecayMul : float
  l2DecayMul : float
  outDepth : int 
  outSx : int
  outSy : int

  switchx : float []
  switchy : float []
} 

type Core =
  | ConvCore of ConvCore
  | FullyConnCore of FullyConnCore
  | DropoutCore of DropoutCore
  | InputCore of InputCore
  | SoftmaxCore of SoftmaxCore
  | RegressionCore of RegressionCore
  | SVMCore of SVMCore
  | ReluCore of ReluCore
  | SigmoidCore of SigmoidCore
  | MaxoutCore of MaxoutCore
  | TanhCore of TanhCore
  | LocalResponseNormalizationCore of LocalResponseNormalizationCore
  | PoolCore of PoolCore
 
type Forward = Core -> Vol -> bool -> Core
type Backward = Core -> Core
type GetParamsAndGrads = Core -> ParamsGrads list


type LayerType = {
  forward : Forward
  backward : Backward
  getParamsAndGrads : GetParamsAndGrads
  data : Core
}
