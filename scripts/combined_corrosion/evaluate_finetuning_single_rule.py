"""Exploratory CV5 of single-rule S2 and matched generic S2 on one dataset.

Steps count additional fine-tuning updates; step zero is the generic 10k parent.
This development comparison is not a frozen final-test evaluation.
"""
import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path

from scripts.combined_corrosion import finetune_single_rule as training
from scripts.epit_pipeline import finetune
from tabicl.train.run_combined_selected_pairs import SOCCOL_OVERRIDE_KEYS

ROOT = training.ROOT
DEFAULT_CONTROL = ROOT/'checkpoints/epit_finetuning/generic10k_generic_s2'
# Historical generic S2 predates selected-pair dispatch and extraction of group
# counts. These exact changes preserve generic generation and optimization.
COMPATIBLE_HASHES = {
    'run.py': frozenset({
        '522b218f5157092eb9df9cf10f066dd0bf481cffc0b4738ebaea6b8fb4023430',
        '40740bbdd8536595ee8466f12757046b7f8e22444e7d12886b845ba7facc4205'}),
    'original_mixture.py': frozenset({
        'ac4ee508973a77fb6fa1686582523732b17f65a762a39196be3b6d124d1ae230',
        'c2f89883b4658d6dc3805d86fc14d05b6ae871d78d3c241532058ed08cc41bce'}),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', choices=['epit', 'soccol', 'combined'], required=True)
    parser.add_argument('--dataset', choices=['epit', 'soccol', 'mpea', 'rebar'], required=True)
    parser.add_argument('--finetune-run', type=Path)
    parser.add_argument('--control-run', type=Path, default=DEFAULT_CONTROL)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output prefix; the dataset name is appended.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--checkpoint', default='all')
    parser.add_argument('--checkpoint-scope', choices=['all', 'common'], default='all')
    parser.add_argument('--checkpoint-step-interval', type=int, default=100)
    parser.add_argument('--max-checkpoint-step', type=int)
    parser.add_argument('--n-estimators', type=int, default=8)
    parser.add_argument('--exploratory-cv-folds', type=int, choices=[5], default=5)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.finetune_run = (args.finetune_run or ROOT/f'checkpoints/{args.prior}_finetuning'/training.NAMES[args.prior]).resolve()
    args.control_run = args.control_run.resolve()
    if args.finetune_run == args.control_run:
        parser.error('Informed and generic S2 must be distinct runs.')
    return args


def comparison_args(args):
    evaluator = importlib.import_module(f'scripts.{args.dataset}_pipeline.evaluate_checkpoint_comparison')
    output = args.output_dir.with_name(f'{args.output_dir.name}_{args.dataset}')
    label = f'{args.prior}_single_rule_s2'
    argv = ['--candidate-run', str(args.finetune_run), '--candidate-label', label,
        '--output-dir', str(output), '--device', args.device, '--checkpoint', args.checkpoint,
        '--checkpoint-scope', args.checkpoint_scope, '--min-checkpoint-step', '0',
        '--checkpoint-step-interval', str(args.checkpoint_step_interval),
        '--exploratory-cv-folds', '5', '--n-estimators', str(args.n_estimators)]
    if args.max_checkpoint_step is not None:
        argv.extend(['--max-checkpoint-step', str(args.max_checkpoint_step)])
    if args.dry_run:
        argv.append('--dry-run')
    options = evaluator.parse_args(argv)
    options._selected_model_runs = [(label, args.finetune_run), ('generic_continuation', args.control_run)]
    return evaluator, options


def load_stage(directory, *, generic=False):
    path = directory/'single_rule_finetuning.json'
    if path.is_file():
        stage = json.loads(path.read_text())
        config = vars(training.validate_plan(stage))
        if (stage['prior'] == 'generic') != generic:
            raise ValueError('Expected a generic control and an informed candidate.')
        identity, stream = stage['finetuning'], stage['generic_stream_identity']
    elif generic:
        stage = json.loads((directory/'finetuning.json').read_text())
        config = vars(finetune.training_config(stage['train_command']))
        identity = stage['overrides']['finetuning']
        stream = stage['manual_pipeline_identity']['fixed_prior']['paired_prior_streams']
        if (config['informed_prior_ratio'] != 0 or config['epit_selected_pair_dir']
                or config['soccol_selected_pair_dir'] or config['epit_selected_pair_informed_only']):
            raise ValueError('Control must continue exclusively on generic tables.')
    else:
        raise FileNotFoundError(f'Missing single-rule S2 metadata: {path}')
    if (identity['schema_version'] != 'epit_finetuning_s2_v1'
            or identity['parent_step'] != 10000 or identity['starting_step'] != 0
            or identity['optimization'] != finetune.S2_SETTINGS
            or identity['table_settings'] != finetune.RETAINED_TABLE_SETTINGS):
        raise ValueError('Expected generic 10k initialization and the established S2 settings.')
    if not (directory/'step-0.ckpt').is_file():
        raise FileNotFoundError(f'Missing unchanged parent at {directory}/step-0.ckpt')
    return stage, config, deepcopy(identity), deepcopy(stream)


def normalize_compatible_hashes(left, right):
    for filename, allowed in COMPATIBLE_HASHES.items():
        a, b = left.get(filename), right.get(filename)
        if a != b and a in allowed and b in allowed:
            left[filename] = right[filename] = 'verified_generic_equivalent_refactor'


def validate_stages(args):
    informed, ci, identity_i, stream_i = load_stage(args.finetune_run)
    control, cg, identity_g, stream_g = load_stage(args.control_run, generic=True)
    if informed['prior'] != args.prior:
        raise ValueError('--prior does not match the saved training run.')
    normalize_compatible_hashes(identity_i['implementation_sha256'], identity_g['implementation_sha256'])
    normalize_compatible_hashes(stream_i['implementation_sha256'], stream_g['implementation_sha256'])
    # Relocated files are equivalent only if their recorded parent hashes match.
    identity_i.pop('parent_checkpoint'); identity_g.pop('parent_checkpoint')
    if identity_i != identity_g or stream_i != stream_g:
        raise ValueError('Candidate and control differ in parent, schedule, stream seed or model/generic implementation.')
    if informed['nproc_per_node'] != control['nproc_per_node']:
        raise ValueError('Candidate and control must use the same GPU count.')
    # These fields affect informed construction only. Generic feature counts are
    # fixed at 2..100 by OriginalPriorMixture, independently of native schemas.
    excluded = SOCCOL_OVERRIDE_KEYS | {
        'checkpoint_dir', 'checkpoint_path', 'wandb_name', 'single_rule_s2_prior', 'combined_soccol_overrides',
        'epit_target_version', 'informed_target_mix_weight', 'informed_target_mix_policy',
        'pitting_coefficient_variation',
    }
    differences = {k for k in set(ci)|set(cg) if k not in excluded and ci.get(k) != cg.get(k)}
    if differences:
        raise ValueError(f'Candidate and control differ beyond informed generation: {sorted(differences)}')
    return {'informed': informed, 'generic_continuation': control}


def run(args):
    evaluator, options = comparison_args(args)
    stages = {} if args.dry_run else validate_stages(args)
    output = evaluator.run(options)
    if not args.dry_run:
        path = output/'results.json'
        payload = json.loads(path.read_text())
        payload.update(finetuning_stages=stages,
            checkpoint_step_semantics='additional S2 updates; step 0 is generic pretraining step 10000',
            primary_comparison_step=2000)
        path.write_text(json.dumps(payload, indent=2)+'\n')
    return output


if __name__ == '__main__':
    run(parse_args())
