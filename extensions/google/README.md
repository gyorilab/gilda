Gilda Google Docs plugin
========================

This plugin allows grounding (i.e., finding identifiers for) bio-entities in
Google Docs documents. The plugin also allows inserting a CURIE representation
of the grounding after the selected entity in the text. The code was partially
adapted from [this tutorial](https://developers.google.com/apps-script/add-ons/editors/docs/quickstart/translate#sidebar.html).

Demo
----

![Gilda Google Docs plugin](gilda_google_docs_v2.gif)

Setup
-----
1. Create a [new Google Doc](https://docs.google.com/document/create).
2. From within your new document, select the menu item *Tools > Script editor*.
   If you are presented with a welcome screen, click *Blank Project*.
3. Rename Code.gs to gilda.gs.
4. Create a new HTML file by clicking Add a file *+ > HTML*. Name this file
   `sidebar` (Apps Script adds the `.html` extension automatically).
5. Replace the contents of `gilda.gs` and `sidebar.html` with the code in this
   folder.
6. Click Save.
7. Click *Untitled project*, and change the name to
   "Gilda named entity normalization," and click *Rename*. The script's name is
   shown to end users in several places, including the authorization dialog.
