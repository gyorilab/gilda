/**
 * @OnlyCurrentDoc
 *
 * Limit the scope to files in which the add-on is used.
 */

function onOpen(e) {
  DocumentApp.getUi().createAddonMenu()
      .addItem('Start', 'showSidebar')
      .addToUi();
}

function onInstall(e) {
  onOpen(e);
}

function showSidebar() {
  const ui = HtmlService.createHtmlOutputFromFile('sidebar')
      .setTitle('Gilda entity normalization');
  DocumentApp.getUi().showSidebar(ui);
}

/**
 * Gets the text the user has selected. If there is no selection,
 * this function displays an error message.
 */
function getSelectedText() {
  const selection = DocumentApp.getActiveDocument().getSelection();
  const text = [];
  if (selection) {
    const elements = selection.getSelectedElements();
    for (let i = 0; i < elements.length; ++i) {
      if (elements[i].isPartial()) {
        const element = elements[i].getElement().asText();
        const startIndex = elements[i].getStartOffset();
        const endIndex = elements[i].getEndOffsetInclusive();

        text.push(element.getText().substring(startIndex, endIndex + 1));
      } else {
        const element = elements[i].getElement();
        // Only ground elements that can be edited as text; skip images and
        // other non-text elements.
        if (element.editAsText) {
          const elementText = element.asText().getText();
          // This check is necessary to exclude images, which return a blank
          // text element.
          if (elementText) {
            text.push(elementText);
          }
        }
      }
    }
  }
  if (!text.length) throw new Error('Please select some text.');
  return text;
}

/**
 * Gets the user-selected text and grounds it.
 */
function getTextAndGrounding() {
  var ui = DocumentApp.getUi();
  const text = getSelectedText().join(' ');
  var term = ground(text)
  var term_summary = '• Name: ' + term['entry_name'] + '\n' 
    + '• Database: ' + term['db'].toLowerCase() + '\n'
    + '• Identifier: ' + term['id']
  var grounding = term['db'].toLowerCase() + ':' + term['id']
  return {
    text: text,
    term: term_summary,
    grounding: grounding
  };
}

/**
 * Inserts grounding text after the selection.
 */
function insertText(grounding) {
  const selection = DocumentApp.getActiveDocument().getSelection();
  const elements = selection.getSelectedElements();

  var element = elements[0]
  var selected_text = element.getElement().asText().getText()

  if (element.isPartial()) {
      var text = selected_text.substring(element.getStartOffset(),
                                         element.getEndOffsetInclusive() + 1);
  } else {
      var text = selected_text
  }

  if (element.isPartial()){
    element.getElement().asText().insertText(element.getEndOffsetInclusive() + 1, ' (' + grounding + ')')
    //.setLinkUrl(0, text.length, response_json[0]['term']['url'])
  } else {
    element.getElement().asText().setText(selected_text + ' (' + grounding + ')')
  }
}

/**
 * Given text, ground it.
 */
function ground(text) {
    options = {
    'method': 'post',
    'contentType': 'application/json',
    'payload': JSON.stringify({'text': text})
  }

  var response = UrlFetchApp.fetch("http://grounding.indra.bio/ground", options);
  var response_json = JSON.parse(response.getContentText())
  if (!response_json.length){
    throw new Error('Couldn\'t ground ' + text);
  }
  var term = response_json[0]['term']
  return term;
}
