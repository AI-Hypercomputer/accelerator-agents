"""CLI entry point: python -m JAXBench <command> [options]

Commands:
    evaluate    Evaluate a candidate kernel against a workload
    run         Run benchmark workloads
    list        List available workloads
"""

import argparse
import json
import sys


def cmd_evaluate(args):
    """Evaluate a candidate kernel."""
    from JAXBench.harness.evaluator import evaluate_kernel, format_eval_result

    result = evaluate_kernel(
        workload_name=args.workload,
        kernel_path=args.kernel,
        tpu=args.tpu,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_eval_result(result))

    return 0 if result['status'] == 'correct' else 1


def cmd_run(args):
    """Run benchmark workloads."""
    from JAXBench.harness.runner import run_workload, run_all

    if args.all:
        output = run_all(
            tpu=args.tpu,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )
        if args.json:
            print(json.dumps(output, indent=2))
        return 0

    if args.workload:
        results = []
        for variant in ['baseline', 'optimized']:
            r = run_workload(
                name=args.workload,
                variant=variant,
                tpu=args.tpu,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
            )
            if r is not None:
                results.append(r)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                if r['status'] == 'success':
                    print(f"{r['name']} ({r['variant']}): "
                          f"{r['median_ms']:.3f}ms, "
                          f"{r['tflops']:.1f} TFLOPS, "
                          f"{r['utilization_pct']:.1f}% util "
                          f"[{r['timing_method']}]")
                else:
                    print(f"{r['name']} ({r['variant']}): ERROR - {r.get('error', '?')}")
        return 0

    print("Error: specify --workload NAME or --all", file=sys.stderr)
    return 1


def cmd_list(args):
    """List available workloads."""
    from JAXBench.benchmark import list_workloads, has_optimized

    workloads = list_workloads()
    if args.json:
        items = [{'name': w, 'has_optimized': has_optimized(w)} for w in workloads]
        print(json.dumps(items, indent=2))
    else:
        print(f"JAXBench: {len(workloads)} workloads\n")
        for w in workloads:
            opt = " [+ optimized]" if has_optimized(w) else ""
            print(f"  {w}{opt}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='JAXBench',
        description='JAXBench: TPU kernel benchmark suite',
    )
    sub = parser.add_subparsers(dest='command')

    # evaluate
    eval_p = sub.add_parser('evaluate', help='Evaluate a candidate kernel')
    eval_p.add_argument('--workload', required=True,
                        help='Workload name (e.g. 1p_Flash_Attention)')
    eval_p.add_argument('--kernel', required=True,
                        help='Path to kernel file with workload() function')
    eval_p.add_argument('--tpu', choices=['v5e', 'v6e', 'auto'], default='auto',
                        help='TPU target (default: auto-detect)')
    eval_p.add_argument('--num-warmup', type=int, default=5,
                        help='Warmup iterations (default: 5)')
    eval_p.add_argument('--num-iters', type=int, default=50,
                        help='Benchmark iterations (default: 50)')
    eval_p.add_argument('--json', action='store_true',
                        help='Output as JSON (for agent consumption)')

    # run
    run_p = sub.add_parser('run', help='Run benchmark workloads')
    run_p.add_argument('--workload', default=None,
                       help='Workload name (run single workload)')
    run_p.add_argument('--all', action='store_true',
                       help='Run all workloads')
    run_p.add_argument('--tpu', choices=['v5e', 'v6e', 'auto'], default='auto',
                       help='TPU target (default: auto-detect)')
    run_p.add_argument('--num-warmup', type=int, default=5,
                       help='Warmup iterations (default: 5)')
    run_p.add_argument('--num-iters', type=int, default=50,
                       help='Benchmark iterations (default: 50)')
    run_p.add_argument('--json', action='store_true',
                       help='Output as JSON')

    # list
    list_p = sub.add_parser('list', help='List available workloads')
    list_p.add_argument('--json', action='store_true',
                        help='Output as JSON')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        'evaluate': cmd_evaluate,
        'run': cmd_run,
        'list': cmd_list,
    }

    return handlers[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
