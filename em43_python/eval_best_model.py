"""
eval_best_model.py - flexible evaluation & visualisation for EM-4/3
===================================================================
- Works with any best_genome.pkl layout (rule/prog pairs or genome key)
- Command-line flags choose input range and step
- Reports:
    - average absolute error
    - success-rate  (|err| < 0.1)
    - accuracy      (% exact matches)
- Generates prediction_plot.png, program_colors.png, optional
  inference_steps_<n>.png

Author: Giacomo Bocchese - with the help of ChatGPT
"""

from __future__ import annotations
import argparse, numpy as np, pickle, matplotlib.pyplot as plt, numba as nb
from matplotlib.colors import ListedColormap
from em43_numba import EM43Batch

nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)
print(f"Numba using {nb.get_num_threads()} threads")

# ───────────── symbols & cmap ─────────────
SYMBOLS = {0: "·", 1: "P", 2: "R", 3: "B"}
CMAP    = ListedColormap(["white", "black", "red", "blue"])

# ───────────── generic loader ─────────────
def load_genome(path="best_genome.pkl"):
    with open(path, "rb") as f:
        d = pickle.load(f)
    if {"rule", "prog"}.issubset(d):
        return (d["rule"], d["prog"]), d.get("fitness", np.nan)
    if "genome" in d:
        return d["genome"], d.get("fitness", np.nan)
    if "best" in d:
        return d["best"][0], float(d["fit"][0])
    raise KeyError("Unrecognised pickle format")

# ───────────── visuals ─────────────
def prog_str(p): return "".join(SYMBOLS[x] for x in p)
def prog_colormap(p):
    plt.figure(figsize=(12,2)); plt.imshow([p], cmap=CMAP, aspect="auto")
    plt.axis("off"); plt.title("Programme"); plt.savefig("program_colors.png", dpi=150, bbox_inches="tight"); plt.close()

def init_state(batch:EM43Batch,n:int):
    L,N=batch.L,batch.N; s=np.zeros(N,np.uint8)
    s[:L]=batch.prog; s[L:L+2]=3; s[L+2+n+1]=2; return s

def vis_steps(batch:EM43Batch,n:int):
    rule=batch.rule; state=init_state(batch,n); frames=[state.copy()]
    for _ in range(batch.max_steps):
        nxt=np.zeros_like(state)
        for i in range(1,batch.N-1):
            idx=(state[i-1]<<4)|(state[i]<<2)|state[i+1]
            nxt[i]=rule[idx]
        frames.append(nxt.copy()); state=nxt
        live=(state!=0).sum()
        if live and (state==3).sum()/live>=batch.halt_thresh: break
    arr=np.stack(frames)
    plt.figure(figsize=(12,0.45*len(arr))); plt.imshow(arr,cmap=CMAP,aspect="auto")
    plt.axis("off"); plt.title(f"Steps for n={n}"); plt.tight_layout()
    fname=f"inference_steps_{n}.png"; plt.savefig(fname,dpi=150,bbox_inches="tight"); plt.close()
    print("Saved",fname)

# ───────────── main ─────────────
def main():
    ap=argparse.ArgumentParser(description="Evaluate best EM-4/3 genome")
    ap.add_argument("-s","--start",type=int,default=1, help="first input (inclusive)")
    ap.add_argument("-e","--end",  type=int,default=30, help="last input (inclusive)")
    ap.add_argument("-t","--step", type=int,default=1, help="step between inputs")
    args=ap.parse_args()

    genome,fit=load_genome()
    batch=EM43Batch(genome,window=200,max_steps=700,halt_thresh=0.5)

    inputs=np.arange(args.start,args.end+1,args.step)
    outs=batch.run(inputs); expected=inputs*2
    errs=np.abs(outs-expected)
    avg_err=float(errs.mean())
    success=float((errs<0.1).mean()*100)
    accuracy=float((outs==expected).mean()*100)

    # console
    print("\nBest EM-4/3 genome")
    print("="*60)
    print(f"Stored fitness : {fit:.3f}")
    print(f"Avg |err|      : {avg_err:.3f}")
    print(f"Success rate  : {success:.1f}%  (|err|<0.1)")
    print(f"Accuracy      : {accuracy:.1f}%  (exact)")
    print("\nProgramme:\n"+prog_str(genome[1]))
    print("\nInput | Out | Exp | Err")
    print("-"*38)
    for n,o,e,er in zip(inputs,outs,expected,errs):
        print(f"{n:5d} | {o:4d} | {e:4d} | {er:4d}")
    # plots
    plt.figure(figsize=(8,6))
    plt.plot(inputs,expected,"b-",lw=2,label="Expected 2×")
    plt.plot(inputs,outs,"r--",lw=2,label="Predicted"); plt.scatter(inputs,outs,color="red")
    plt.grid(alpha=0.3); plt.xlabel("Input"); plt.ylabel("Output"); plt.title("Expected vs Predicted")
    plt.legend(); plt.tight_layout(); plt.savefig("prediction_plot.png",dpi=150); plt.close()
    prog_colormap(genome[1])

    while True:
        try:
            val=int(input("\nEnter n in range to visualise steps (0=exit): "))
            if val==0: break
            if val in inputs:
                vis_steps(batch,val)
            else:
                print("Not in the evaluated range.")
        except ValueError:
            print("Enter an integer.")

if __name__=="__main__":
    main()
