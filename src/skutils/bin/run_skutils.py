# Standard library
import argparse
from datetime import datetime
import logging
from pathlib import Path
import pprint
import shutil

# 3rd party
import sklearn
import yaml

# 1st party
from skutils import git, logging_utils, scripting
from skutils.bin import fit_supervised_model

# Globals
log = logging.getLogger(Path(__file__).stem)

# Configuration
sklearn.set_config(transform_output='pandas')

################################################################################
def main():
    args = parse_argv()
    cfg  = args.get_config(args)

    ocfg = cfg['outputs']
    odir = Path(ocfg['path'])
    cfg_save = ocfg['toggles']

    # Setup
    if ocfg["timestamp_subdir"]:
        odir.mkdir(exist_ok=True)
        odir = odir / datetime.now().strftime("%Y%m%d_%H%M%S")
    scripting.require_empty_dir(odir, overwrite=ocfg['overwrite'])
    logging_utils.update_log_filenames(odir, cfg['logging'])
    logging_utils.configure_logging(**cfg['logging'])
    logging_utils.require_root_console_handler(args.log_cli_level)
    log.debug("Logging Summary:\n%s", logging_utils.summarize_logging())

    # Reproducibility
    log.debug("Final configuration:\n%s", pprint.pformat(cfg, indent=4))
    if cfg_save['save_input_configs']:
        buf = len(args.configs)
        for i, path in enumerate(args.configs):
            opath = odir / f'config_input{i:0{buf}d}_{Path(path).stem}.yml'
            shutil.copyfile(path, opath)
            log.info("Input configuration saved: %s", opath)
    if cfg_save['save_final_config']:
        opath = odir / "config.yml"
        yaml.safe_dump(cfg, opath.open("w"))
        log.info("Final configuration saved: %s", opath)

    if (working_dir := git.find_working_dir(Path(__file__))) is not None:
        log.debug("Version Control Summary:\n%s", git.summarize_version_control(working_dir))
        if cfg_save['save_git_diff']:
            opath = odir / "git_diff.patch"
            opath.write_text(git.get_diff(working_dir))
            log.info("Git diff patch saved: %s", opath)

    args.main(cfg)

def parse_argv():
    parser = argparse.ArgumentParser()
    skutils = parser.add_subparsers()

    ################################################################################
    # Supervised learning routines
    # >> skutils supervised
    ################################################################################
    supervised = skutils.add_parser(
        name    = 'supervised',
        aliases = ['sup'],
        help    = 'Supervised learning routines',
    ).add_subparsers()

    ########################################
    # >> skutils supervised run
    run = supervised.add_parser(
        name = 'run',
        help = 'Run supervised learning end-to-end'
    )
    fit_supervised_model.add_arguments(run, start_stage="run")
    run.set_defaults(
        main = fit_supervised_model.main,
        get_config = fit_supervised_model.get_config,
    )

    ########################################
    # skutils supervised score
    score = supervised.add_parser(
        name = 'score',
        help = '(TODO) Run only scoring stage'
    )
    fit_supervised_model.add_arguments(score, start_stage="score")
    score.set_defaults(
        main = fit_supervised_model.main,
        get_config = fit_supervised_model.get_config,
    )

    ########################################
    # skutils supervised visualize
    visualize = supervised.add_parser(
        name = 'visualize',
        help = '(TODO) Run only visualization stage'
    )
    fit_supervised_model.add_arguments(visualize, start_stage="visualize")
    visualize.set_defaults(
        main = fit_supervised_model.main,
        get_config = fit_supervised_model.get_config,
    )

    ################################################################################
    # Unsupervised learning routines
    # >> skutils unsupervised
    ################################################################################
    unsupervised = skutils.add_parser(
        name    = 'unsupervised',
        aliases = ['unsup'],
        help    = 'Unsupervised learning routines',
    ).add_subparsers()

    ########################################
    # >> skutils unsupervised run
    run = unsupervised.add_parser(
        name = 'run',
        help = '(TODO) Run unsupervised learning end-to-end'
    )
    run.add_argument('-i', '--input', metavar='PATH')
    run.add_argument('-o', '--odir', metavar='PATH')

    ################################################################################
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
