import sys
import getopt
import logging

import train as train_script
import predict as predict_script
import scrapper as scrapper_script

logging.basicConfig(level=logging.DEBUG)

def main(argv):
    try:
        opts, _ = getopt.getopt(argv, "st:p:")
        if len(opts) != 1:
            raise getopt.GetoptError('Bad argument')

        opt, _ = opts[0]
        if opt == '-s':
            scrapper_script.run(logging)
        elif opt == '-t':
            if len(argv) != 2:
                raise getopt.GetoptError('Bad argument')

            train_script.run(logging, argv[1])
        elif opt == '-p':
            if len(argv) != 3:
                raise getopt.GetoptError('Bad argument')
        
            entities = predict_script.run(logging, argv[1], argv[2])
            for e in entities:
                print(e)
        else:
            raise getopt.GetoptError('Bad argument')
    except getopt.GetoptError:
        print('Usage: main.py [-s] [-t "model name"] [-p "model name" "Text here"]')
        print(opts)
        print(argv)
        exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])