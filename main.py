"""
Deep Learning Framework
Version 1.0
Authors: Benoit Vuillemin, Frederic Bertrand
Licence: AGPL v3
"""

from Managers.common_functions import *
from defaults import *
from config import *
from Managers.encoder_manager import EncoderManager

if __name__ == '__main__':
    # Create the orchestrator and the data preparator
    if orchestrator_from_file:
        orchestrator = load_orchestrator_from_file(output_name)
    else:
        # Concatenate data and co-variable encoders
        if consider_cov:
            # Automatically generate co-variable encoders if needed
            if auto_build_cov_encoders:
                skip = 4 if double_timestamps else 3
                cov_encoders = auto_cov_encoders(input_path, input_chunk_size, skip)
            encoders += cov_encoders
        encoder_manager = EncoderManager(encoders)
        editor_manager = EditorManager(editors)
        orchestrator = build_orchestrator(input_path, output_name, input_chunk_size, encoder_manager,
                                          editor_manager, dates_ids, double_timestamps)
        orchestrator.save_to_file()
    preparator.build(input_chunk_size, output_chunk_size, batch_size, orchestrator)

    # ONLINE
    if mode == "online":
        trainer.build(preparator, epoch_counter)
        model = trainer.train_model_online()

    # OFFLINE
    if mode == "offline":
        if "all" in offline_steps or "encode" in offline_steps:
            orchestrator.process_offline(input_chunk_size, False, debug=debug)
        if "all" in offline_steps or "decode" in offline_steps:
            decode_offline(output_name, output_chunk_size)
        if "all" in offline_steps or "prepare" in offline_steps:
            preparator.run_offline()
        # It if possible to create a new data preparator afterwards and affect it to the LSTM model
        if "all" in offline_steps or "train" in offline_steps:
            trainer.build(preparator, epoch_counter)
            model = trainer.train_model_offline()

    # EDIT DATABASE
    if mode == "edit_db":
        if cov_path:
            orchestrator.process_offline(input_chunk_size, True, cov_path)
        else:
            orchestrator.process_offline(input_chunk_size, True)
