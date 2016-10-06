let rndg = System.Random()

let rec gaussRandom () =
  let u = 2.0 * rndg.NextDouble() - 1.0
  let v = 2.0 * rndg.NextDouble() - 1.0
  let r = v*v + u*u
  
  if r = 0.0 || r > 1.0 then gaussRandom()
  else 
    let c = sqrt ( -2.0 * (log r) / r)
    u * c
  
let randf a b = rndg.NextDouble() * (b - a) + a
let randi a b = randf a b |> int |> float
let randn mu std = mu + gaussRandom() * std 

let zeros = Array.zeroCreate<float>

let arrContains = Array.contains


