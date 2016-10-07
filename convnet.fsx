type T = float * float
type Vol = T [,,]

let emptyVol:Vol = Array3D.zeroCreate<T> 0 0 0

type ParamsGrads = {
  params : float
  grads: float
  l1DecayMul : float
  l2DecayMul : float
}

type Forward<'Core> = 'Core -> Vol -> bool -> 'Core
type Backward<'Core> = 'Core -> 'Core
type GetParamsAndGrads<'Core> = 'Core -> ParamsGrads list



type InputCore = {
  forward : Forward<InputCore>
  backward : Backward<InputCore>
  getParamsAndGrads : GetParamsAndGrads<InputCore>
  outDepth : int
  outSx : int
  outSy : int
  inAct : Vol
  outAct : Vol  
} 

type ConvCore = {
  forward : Forward<ConvCore>
  backward : Backward<ConvCore>
  getParamsAndGrads : GetParamsAndGrads<ConvCore>
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
} 

type FullyConnCore = {
  forward : Forward<FullyConnCore>
  backward : Backward<FullyConnCore>
  getParamsAndGrads : GetParamsAndGrads<FullyConnCore>
  outDepth : int
  l1DecayMul : float
  l2DecayMul : float
  numInputs : int
  outSx : int
  outSy : int
  filters : Vol list
  biases : Vol
} 

type DropoutCore = {
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list

  outSx : int
  outSy : int

  outDepth : int 
  dropProb : float
  dropped : float []
  inAct : Vol
  outAct : Vol
} 

type SoftmaxCore = {
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list


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
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list


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
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list


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
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list

  outSx : int
  outSy : int

  outDepth : int 
  inAct : Vol
  outAct : Vol
} 

type SigmoidCore = {
  forward : Forward<SigmoidCore>
  backward : Backward<SigmoidCore>
  getParamsAndGrads : GetParamsAndGrads<SigmoidCore>

  outSx : int
  outSy : int

  outDepth : int 
  inAct : Vol
  outAct : Vol
} 

type MaxoutCore = {
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list

  outSx : int
  outSy : int
  groupSize : int
  outDepth : int 

  switches : float []

  inAct : Vol
  outAct : Vol
} 

type TanhCore = {
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list

  outSx : int
  outSy : int

  outDepth : int 
  inAct : Vol
  outAct : Vol
} 

type LocalResponseNormalizationCore = {
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list


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
  forward : Forward
  backward : Backward
  getParamsAndGrads : unit -> ParamsGrads list

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

type LayerType = 
  | ConvLayer of ConvCore
  | FullyConnLayer of FullyConnCore
  | DropoutLayer of DropoutCore
  | InputLayer of InputCore
  | SoftmaxLayer of SoftmaxCore
  | RegressionLayer of RegressionCore
  | SVMLayer  of SVMCore
  | ReluLayer of ReluCore
  | SigmoidLayer  of SigmoidCore
  | MaxoutLayer  of MaxoutCore
  | TanhLayer of TanhCore
  | LocalResponseNormalizationLayer of LocalResponseNormalizationCore
  | PoolLayer of PoolCore

