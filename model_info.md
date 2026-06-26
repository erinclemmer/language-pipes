#### model_info[[huggingface_hub.HfApi.model_info]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.21.0.rc0/src/huggingface_hub/hf_api.py#L3242)

Get info on one specific model on huggingface.co

Model can be private if you pass an acceptable token or are logged in.

> [!TIP]
> Raises the following errors:
>
>     - [RepositoryNotFoundError](/docs/huggingface_hub/v1.21.0.rc0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
>       If the repository to download from cannot be found. This may be because it doesn't exist,
>       or because it is set to `private` and you do not have access.
>     - [RevisionNotFoundError](/docs/huggingface_hub/v1.21.0.rc0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)
>       If the revision to download from cannot be found.

**Parameters:**

repo_id (`str`) : A namespace (user or an organization) and a repo name separated by a `/`.

revision (`str`, *optional*) : The revision of the model repository from which to get the information.

timeout (`float`, *optional*) : Whether to set a timeout for the request to the Hub.

securityStatus (`bool`, *optional*) : Whether to retrieve the security status from the model repository as well. The security status will be returned in the `security_repo_status` field.

files_metadata (`bool`, *optional*) : Whether or not to retrieve metadata for files in the repository (size, LFS metadata, etc). Defaults to `False`.

expand (`list[ExpandModelProperty_T]`, *optional*) : List properties to return in the response. When used, only the properties in the list will be returned. This parameter cannot be used if `securityStatus` or `files_metadata` are passed. Possible values are `"author"`, `"baseModels"`, `"cardData"`, `"childrenModelCount"`, `"config"`, `"createdAt"`, `"disabled"`, `"downloads"`, `"downloadsAllTime"`, `"evalResults"`, `"gated"`, `"gguf"`, `"inference"`, `"inferenceProviderMapping"`, `"lastModified"`, `"library_name"`, `"likes"`, `"mask_token"`, `"model-index"`, `"pipeline_tag"`, `"private"`, `"safetensors"`, `"sha"`, `"siblings"`, `"spaces"`, `"tags"`, `"transformersInfo"`, `"trendingScore"`, `"widgetData"`, `"usedStorage"`, and `"resourceGroup"`.

token (`bool` or `str`, *optional*) : A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see https://huggingface.co/docs/huggingface_hub/quick-start#authentication). To disable authentication, pass `False`.

**Returns:**

`[huggingface_hub.hf_api.ModelInfo](/docs/huggingface_hub/v1.21.0.rc0/en/package_reference/hf_api#huggingface_hub.ModelInfo)`

The model repository information.