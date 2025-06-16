from em43_utilis import get_args
from em43_ga import GenomeAlgorithm
from em43_class import EM43
import time
import numba as nb

t0 = time.time()
args = get_args()
stage = args.stage
nb.set_num_threads(nb.config.NUMBA_NUM_THREADS)

if stage == "train":
    print("\n------------------------------")
    print(f"Numba using {nb.get_num_threads()} threads")
    print("Starting EM-4/3 GA with parameters:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")
    print("------------------------------\n")

    ga = GenomeAlgorithm(args)
    best_rule, best_prog, best_fitness = ga.run()

    print("\n------------------------------")    
    print(f"Elapsed {time.time()-t0:.1f}s")
    print(f"Best fitness: {best_fitness:.3f}")
    print(f"Best rule: {best_rule}")
    print(f"Best program: {best_prog}")
    print("\n------------------------------")

    print("\n------------------------------")
    print("\nInfer from the best genome...")
    em43 = EM43(best_rule, best_prog)
    em43.infer()
    print("\n------------------------------")
    print("\nEvaluate from the best genome...")
    em43.evaluate()
    print("\n------------------------------")
    

if stage == "infer":
    print("\n------------------------------")
    print("\nInfer from the best genome by loading it from best_genome.pkl...")
    em43 = EM43()
    em43.load_genome()
    em43.infer()
    print("\n------------------------------")
    print("\nEvaluate from the best genome...")
    em43.evaluate()
    print("\n------------------------------")

if stage == "evaluate":
    print("\n------------------------------")
    print("\nEvaluate from the best genome by loading it from best_genome.pkl...")
    em43 = EM43()
    em43.load_genome()
    em43.evaluate()
    print("\n------------------------------")   