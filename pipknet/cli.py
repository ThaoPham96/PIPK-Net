import argparse


def _print_summary(summary, title):
    print(f"\n--- PIPK-Net Ensemble Summary [{title}] ---")
    print(f"{'Parameter':<15} | {'Mean':<10} | {'Std Dev':<10}")
    print("-" * 40)
    for param, row in summary.iterrows():
        print(f"{param:<15} | {row['Mean']:<10.4f} | {row['Std']:<10.4f}")


def main():
    parser = argparse.ArgumentParser(description="PIPK-Net Command Line Tool")
    subparsers = parser.add_subparsers(dest="command")

    # --- TRAIN ---
    train_parser = subparsers.add_parser("train", help="Nested 5-fold scaffold CV training")
    train_parser.add_argument("--data", type=str, required=True,
                              help="CSV with a SMILES column, IonType, and raw PK columns")
    train_parser.add_argument("--variant", choices=["A_baseline", "B_ion", "C_physio"], required=True)
    train_parser.add_argument("--out", type=str, default="checkpoints",
                              help="Output directory for per-fold checkpoints")
    train_parser.add_argument("--epochs", type=int, default=200, help="Max epochs per fit")
    train_parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience")
    train_parser.add_argument("--quick", action="store_true",
                              help="Use a tiny hyperparameter grid (smoke test, not for publication)")

    # --- PREDICT ---
    predict_parser = subparsers.add_parser("predict", help="Predict PK for a SMILES or a CSV file")
    group = predict_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles", type=str, help="Single SMILES string")
    group.add_argument("--file", type=str, help="CSV file with 'SMILES' (and optional 'IonType', 'Name') columns")
    predict_parser.add_argument("--ion", default="neutral",
                                choices=["anionic", "cationic", "neutral", "zwitterionic"],
                                help="Ionisation state for single-SMILES mode (default: neutral)")
    predict_parser.add_argument("--weights", type=str, required=True, help="Path to a variant folder")
    predict_parser.add_argument("--output", type=str, default="predictions.csv",
                                help="Output CSV path (batch mode)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "train":
        from pipknet.training import train_from_csv
        print(f"Training variant '{args.variant}' from {args.data} -> {args.out}")
        train_from_csv(args.data, args.variant, args.out, quick=args.quick,
                       max_epochs=args.epochs, patience=args.patience)
        print(f"Done. Checkpoints written under {args.out}/{args.variant}/")
        return

    if args.command == "predict":
        from pipknet.inference import PIPKNetPredictor
        predictor = PIPKNetPredictor(args.weights)

        if args.smiles:
            summary = predictor.predict(args.smiles, ion_type=args.ion)
            _print_summary(summary, predictor.variant)

        elif args.file:
            print(f"\nLoading drugs from: {args.file}")
            results = predictor.predict_batch(args.file)
            results.to_csv(args.output, index=False)
            if len(results) <= 10:
                for _, r in results.iterrows():
                    summary_cols = {c[:-5]: r[c] for c in results.columns if c.endswith("_mean")}
                    print(f"\n[{r.get('Name', '')}]")
                    for param, mean in summary_cols.items():
                        std = r[f"{param}_std"]
                        print(f"  {param:<15} {mean:>10.4f} +/- {std:<8.4f}")
            print(f"\n{'=' * 60}\nDONE: {len(results)} drugs -> {args.output}\n{'=' * 60}")


if __name__ == "__main__":
    main()
