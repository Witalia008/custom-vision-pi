# Using Event Grid trigger instead of Blob Storage trigger
## (because size is limited to 128 Mb)

* Full instructions:
[Azure Event Grid trigger for Azure Functions](https://docs.microsoft.com/en-us/azure/azure-functions/functions-bindings-event-grid-trigger?tabs=python)

* Create 'Event Grid Trigger'-based Azure Function in VS Code
* Deploy to Azure
* Set `stovedatastorage_STORAGE` env variable in `local.settings.json` with the storage connection string.
* In Azure Function in the Portal, in the code window, click "Add Event Grid subscription":
  - Add subject filter:
    - Prefix: `/blobServices/default/containers/{container name}/blobs`
    - Suffix: `.mp4`
* Use POST requests for testing. Sample body below.
* Use blob sdk to download/upload files:
  [Azure Storage Blobs client library for Python](https://github.com/Azure/azure-sdk-for-python/tree/master/sdk/storage/azure-storage-blob)


## Sample request body for local testing.
POST:
```
http://localhost:7071/runtime/webhooks/eventgrid?functionName=<function name>
```
Headers:
```
Content-Type: application/json
aeg-event-type: Notification
```
Body:
```
[{
  "topic": "/subscriptions/{subscriptionid}/resourceGroups/eg0122/providers/Microsoft.Storage/storageAccounts/egblobstore",
  "subject": "/blobServices/default/containers/{containername}/blobs/blobname.mp4",
  "eventType": "Microsoft.Storage.BlobCreated",
  "eventTime": "2018-01-23T17:02:19.6069787Z",
  "id": "{guid}",
  "data": {
    "api": "PutBlockList",
    "clientRequestId": "{guid}",
    "requestId": "{guid}",
    "eTag": "0x8D562831044DDD0",
    "contentType": "video/mp4",
    "contentLength": 2248,
    "blobType": "BlockBlob",
    "url": "https://egblobstore.blob.core.windows.net/{containername}/blobname.mp4",
    "sequencer": "000000000000272D000000000003D60F",
    "storageDiagnostics": {
      "batchId": "{guid}"
    }
  },
  "dataVersion": "",
  "metadataVersion": "1"
}]
```