"""Initial dependency extraction; refuses an existing destination.

HISTORICAL TOOL: do not rerun for shared-core development; edit src/Bayesian_state.
Audited post-extraction adjustments are documented in Bayesian_model/MODEL_0826_AUDIT.md.
The current package and final manifest, not this extraction alone, define the delivered version."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

OLD='src.Bayesian_state'
NEW='CategoryLearning_codes.Bayesian_model'
SOURCE=Path('src/Bayesian_state')
DEST=Path('CategoryLearning_codes/Bayesian_model')
OVERRIDES={
 'model/modules/hypothesis_transition/__init__.py': '''"""Model 0826 transition and shared contracts."""
from .contracts import HypothesisSelection, HypothesisTransitionResult, TransitionContext, TwoStepHypothesisTransitionMixin
from .nested_feedback_accumulator import NestedFeedbackAccumulatorHypothesisTransitionModule
''',
 'model/modules/__init__.py': '''"""Cognitive modules used by Model 0826 and its P/PM/PH/PMH cells."""
from .base_module import BaseModule, ModulePhase, ModuleRole
from .beta import BetaModule
from .memory import BayesianMemoryModule, DualMemoryModule
from .perception import PerceptionModule
from .hypothesis_transition import NestedFeedbackAccumulatorHypothesisTransitionModule
''',
 'optimization/__init__.py':'"""Model 0826 parameter search and diagnostics; import explicit submodules."""\n',
 'evaluation/__init__.py':'"""Model 0826 validation and report alignment; import explicit submodules."""\n',
}
ROOTS=[
 'model/state_model.py','model/modules/perception.py','model/modules/beta.py',
 'model/modules/memory.py','model/modules/hypothesis_transition/nested_feedback_accumulator.py',
 'inference/dispatcher.py','inference/posterior_predictive.py',
 'optimization/model_0826.py','optimization/cli.py','optimization/seed_convergence.py',
 'evaluation/model_recovery.py','evaluation/autonomous_trajectories.py',
 'evaluation/internal_cognitive_trajectories.py',
 'run_simulation.py','run_hyper_then_simulation.py','run_hyper_evaluation.py',
 'run_autonomous_trajectory_evaluation.py','run_internal_cognitive_trajectory_evaluation.py',
]
ROOTS += [str(p.relative_to(SOURCE)) for folder in ['evaluation/oral','evaluation/particle_filter'] for p in (SOURCE/folder).glob('*.py')]


def main():
    if DEST.exists():raise FileExistsError(DEST)
    files={OLD+'.'+str(p.relative_to(SOURCE)).removesuffix('.py').replace('/','.') .removesuffix('.__init__'):p for p in SOURCE.rglob('*.py')}
    files[OLD]=SOURCE/'__init__.py'
    pending=list(ROOTS);selected={};reasons={r:'0826 workflow root' for r in ROOTS}
    def enqueue(module,why):
        if module not in files:return
        r=str(files[module].relative_to(SOURCE))
        if r not in selected and r not in pending:pending.append(r);reasons[r]=why
    while pending:
        rel=pending.pop(0)
        if rel in selected:continue
        path=SOURCE/rel
        text=OVERRIDES.get(rel,path.read_text())
        selected[rel]=text
        module=OLD+'.'+rel.removesuffix('.py').replace('/','.').removesuffix('.__init__')
        package=module if rel.endswith('__init__.py') else module.rsplit('.',1)[0]
        for parent in path.relative_to(SOURCE).parents:
            init=SOURCE/parent/'__init__.py'
            if init.exists():enqueue(OLD+('' if str(parent)=='.' else '.'+str(parent).replace('/','.')),rel)
        for node in ast.walk(ast.parse(text)):
            if isinstance(node,ast.Import):
                for name in node.names:enqueue(name.name,rel)
            elif isinstance(node,ast.ImportFrom):
                prefix=node.module or ''
                if node.level:
                    prefix='.'.join(package.split('.')[:len(package.split('.'))-node.level+1])+('.'+prefix if prefix else '')
                enqueue(prefix,rel)
                for name in node.names:enqueue(prefix+'.'+name.name,rel)
    DEST.mkdir(parents=True)
    records=[]
    for rel,text in sorted(selected.items()):
        dest=DEST/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(text.replace(OLD,NEW))
        records.append({'file':rel,'source_sha256':hashlib.sha256((SOURCE/rel).read_bytes()).hexdigest(),
                        'reason':reasons.get(rel,'package ancestor'),'export_pruned':rel in OVERRIDES})
    resource='hypothesis_space/resources/similarity/similarity_matrix_shared_hypothesis_space_v1_d4_c2_n100000_pairtol0p1_centertol0p1.npy'
    (DEST/resource).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/resource,DEST/resource)
    excluded=[str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*.py') if str(p.relative_to(SOURCE)) not in selected]
    manifest={'manuscript':'manuscript/model_0826.tex','manuscript_sha256':hashlib.sha256(Path('manuscript/model_0826.tex').read_bytes()).hexdigest(),
              'roots':ROOTS,'included':records,'excluded_python':sorted(excluded),'resource':resource,
              'resource_sha256':hashlib.sha256((DEST/resource).read_bytes()).hexdigest()}
    (DEST/'MIGRATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print('Included',len(records),'Excluded',len(excluded))
    print('Excluded',excluded)

if __name__=='__main__':main()
