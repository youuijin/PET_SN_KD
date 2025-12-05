from Trainer.Trainer_KD_field import Trainer_KD_field

def set_trainer(args):
    if args.KD == "field":
        trainer = Trainer_KD_field(args)
    else:
        raise ValueError("Error! Undefined Method:", args.method)
    
    return trainer