import torch


class Model:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # get the encoded input and the index of the masked token in the input
    def encode_text(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        mask_token_index = torch.where(encoded_input['input_ids'] == self.tokenizer.mask_token_id)[1].item()
        return encoded_input, mask_token_index

    # get indices of the label and the distractor relative to the vocabulary
    def get_indices(self, input_label1, input_label2, input_distractor1, input_distractor2):
        label_index1 = self.tokenizer.encode(input_label1, add_special_tokens=False)[0]
        label_index2 = self.tokenizer.encode(input_label2, add_special_tokens=False)[0]
        distractor_index1 = self.tokenizer.encode(input_distractor1, add_special_tokens=False)[0]
        distractor_index2 = self.tokenizer.encode(input_distractor2, add_special_tokens=False)[0]
        return label_index1, label_index2, distractor_index1, distractor_index2

    # return the predictions of the masked token over the vocabulary
    def predict_probabilities(self, encoded_input):
        output = self.model(**encoded_input)
        predictions = output.logits
        return predictions

    # return the most probable token
    def predict_most_probable_token(self, predictions, mask_token_index):
        masked_token_predictions = predictions[0, mask_token_index]
        top_predicted_token = masked_token_predictions.argmax(axis=-1).item()
        decode_top_predicted_token = self.tokenizer.decode(top_predicted_token)
        return decode_top_predicted_token

    # return the most probable token among the label and the distractor and softmax probabilities
    def predict_label_distractor_token(self, predictions, mask_token_index, label_index1, label_index2,
                                       distractor_index1, distractor_index2, values):
        masked_token_predictions = predictions[0, mask_token_index]
        masked_token_predictions_label_distractor = masked_token_predictions[
            [label_index1, label_index2, distractor_index1, distractor_index2]]
        predictions_softmax = torch.softmax(masked_token_predictions_label_distractor, dim=0)
        predicted_token_index = torch.argmax(predictions_softmax).item()
        predicted_token = values[predicted_token_index]
        return predicted_token, predictions_softmax

    #
    def predict_masked_token(self, text, return_label_distractor_prob=True, return_most_prob=True):
        with torch.no_grad():
            text, input_label1, input_label2, input_distractor1, input_distractor2 = text['prompt'], text['label1'], \
                text['label2'], text['distractor1'], text['distractor2']
            values = [input_label1, input_label2, input_distractor1, input_distractor2]

            encoded_input, mask_token_index = self.encode_text(text)
            label_index1, label_index2, distractor_index1, distractor_index2 = self.get_indices(input_label1,
                                                                                                input_label2,
                                                                                                input_distractor1,
                                                                                                input_distractor2)
            predictions = self.predict_probabilities(encoded_input)

            returns = []

            predicted_token, predictions_softmax = self.predict_label_distractor_token(predictions, mask_token_index,
                                                                                       label_index1, label_index2,
                                                                                       distractor_index1,
                                                                                       distractor_index2, values)
            returns.insert(0, predicted_token)

            if return_label_distractor_prob:
                returns.append(predictions_softmax[0] + predictions_softmax[1])
                returns.append(predictions_softmax[2] + predictions_softmax[3])

            if return_most_prob:
                most_probable_token = self.predict_most_probable_token(predictions, mask_token_index)
                returns.append(most_probable_token)

            return returns

    # def predict_masked_token(self, text):
    #     with torch.no_grad():
    #         text, input_label1, input_label2, input_distractor1, input_distractor2 = text['prompt'], text['label1'], text[
    #             'label2'], text['distractor1'], text['distractor2']
    #
    #         # Encode the text
    #         encoded_input = self.tokenizer(text, return_tensors='pt')
    #         mask_token_index = torch.where(encoded_input['input_ids'] == self.tokenizer.mask_token_id)[1].item()
    #         # mask_token_index = (encoded_input.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    #
    #         # Get the label and distractor indices
    #         values = [input_label1, input_label2, input_distractor1, input_distractor2]
    #         label_index1 = self.tokenizer.encode(input_label1, add_special_tokens=False)[0]
    #         label_index2 = self.tokenizer.encode(input_label2, add_special_tokens=False)[0]
    #         distractor_index1 = self.tokenizer.encode(input_distractor1, add_special_tokens=False)[0]
    #         distractor_index2 = self.tokenizer.encode(input_distractor2, add_special_tokens=False)[0]
    #
    #         # Predict the masked token
    #         output = self.model(**encoded_input)
    #         predictions = output.logits
    #         masked_token_predictions = predictions[0, mask_token_index]
    #         masked_token_predictions_softmax = torch.softmax(masked_token_predictions, dim=0)
    #         masked_token_predictions_softmax_label_distractor = masked_token_predictions_softmax[
    #             [label_index1, label_index2, distractor_index1, distractor_index2]]
    #
    #         top_predicted_token = masked_token_predictions_softmax.argmax(axis=-1).item()
    #         decode_top_predicted_token = self.tokenizer.decode(top_predicted_token)
    #         # Get the predicted token index
    #         predicted_token_index = torch.argmax(masked_token_predictions_softmax_label_distractor).item()
    #         predicted_token = values[predicted_token_index]
    #         return predicted_token, decode_top_predicted_token
    #
