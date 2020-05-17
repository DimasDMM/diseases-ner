import logging
from lib.InitialPreprocess import InitialPreprocess
from lib.FileManager import FileManager
from lib.NlpTool import NlpTool

def run(logging):
    nlp_tool = NlpTool()
    file_manager = FileManager()
    ip = InitialPreprocess(nlp_tool)
    
    logging.info('Loading dataset...')
    train_dataset, test_dataset = file_manager.getRawDataset()

    logging.info('Removing initial number...')
    train_dataset = ip.apply_initial_number_remover(train_dataset)
    test_dataset = ip.apply_initial_number_remover(test_dataset)

    logging.info('Spliting dataset by sentences...')
    train_dataset = ip.apply_sentences_split(train_dataset)
    test_dataset = ip.apply_sentences_split(test_dataset)

    logging.info('Getting entities in tags...')
    train_entities = ip.get_entities_in_category_tags(train_dataset)
    test_entities = ip.get_entities_in_category_tags(test_dataset)

    logging.info('Removing category tags from dataset...')
    train_dataset = ip.apply_tag_cleaner(train_dataset)
    test_dataset = ip.apply_tag_cleaner(test_dataset)

    logging.info('Computing dataset as binary strings (it may take a few minutes)...')
    bin_train_dataset = ip.get_bin_dataset(train_dataset, train_entities)
    bin_test_dataset = ip.get_bin_dataset(test_dataset, test_entities)

    bin_train_dataset = [' '.join(t.astype('str')) for t in bin_train_dataset]
    bin_test_dataset = [' '.join(t.astype('str')) for t in bin_test_dataset]
    
    logging.debug('Saving dataset...')
    file_manager.saveParsedDataset(train_dataset, test_dataset, bin_train_dataset, bin_test_dataset)

    logging.info('Done!')

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info('## INITIAL DATASET PREPROCESS ##')
    run(logging)
