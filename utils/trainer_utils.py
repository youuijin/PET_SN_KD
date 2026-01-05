from Trainer.Trainer_KD_field import Trainer_KD_field
from Trainer.Trainer_direct import Trainer_direct

def set_trainer(args):
    if args.KD == "field":
        trainer = Trainer_KD_field(args)
    elif args.KD == 'None':
        trainer = Trainer_direct(args)
    else:
        raise ValueError("Error! Undefined Method:", args.method)
    
    return trainer