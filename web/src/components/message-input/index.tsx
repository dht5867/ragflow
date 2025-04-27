import { useTranslate } from '@/hooks/common-hooks';
import {
  useDeleteDocument,
  useFetchDocumentInfosByIds,
  useRemoveNextDocument,
  useUploadAndParseDocument,
} from '@/hooks/document-hooks';
import { cn } from '@/lib/utils';
import { getExtension } from '@/utils/document-util';
import { formatBytes } from '@/utils/file-util';
import {
  CloseCircleOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
} from '@ant-design/icons';
import type { GetProp, UploadFile } from 'antd';
import {
  Button,
  Card,
  Divider,
  Flex,
  Input,
  List,
  Popover,
  Space,
  Spin,
  Typography,
  Upload,
  UploadProps,
  notification,
} from 'antd';
import get from 'lodash/get';
import { CircleStop, Paperclip, SendHorizontal } from 'lucide-react';
import {
  ChangeEvent,
  ChangeEventHandler,
  memo,
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react';
import FileIcon from '../file-icon';
import styles from './index.less';

type FileType = Parameters<GetProp<UploadProps, 'beforeUpload'>>[0];
const { Text } = Typography;

const { TextArea } = Input;

const getFileId = (file: UploadFile) => get(file, 'response.data.0');

const getFileIds = (fileList: UploadFile[]) => {
  const ids = fileList.reduce((pre, cur) => {
    return pre.concat(get(cur, 'response.data', []));
  }, []);

  return ids;
};

const isUploadSuccess = (file: UploadFile) => {
  const code = get(file, 'response.code');
  return typeof code === 'number' && code === 0;
};

interface IProps {
  disabled: boolean;
  value: string;
  sendDisabled: boolean;
  sendLoading: boolean;
  onPressEnter(documentIds: string[]): void;
  onInputChange: ChangeEventHandler<HTMLTextAreaElement>;
  conversationId: string;
  uploadMethod?: string;
  isShared?: boolean;
  showUploadIcon?: boolean;
  createConversationBeforeUploadDocument?(message: string): Promise<any>;
  stopOutputMessage?(): void;
  onSelect: ChangeEventHandler<HTMLInputElement>;

}

const getBase64 = (file: FileType): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file as any);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });

const MessageInput = ({
  isShared = false,
  disabled,
  value,
  onPressEnter,
  sendDisabled,
  sendLoading,
  onInputChange,
  conversationId,
  showUploadIcon = true,
  createConversationBeforeUploadDocument,
  uploadMethod = 'upload_and_parse',
  stopOutputMessage,
  onSelect, // 添加 onSelect 回调属性
}: IProps) => {
  const { t ,i18n} = useTranslate('chat');
  const { removeDocument } = useRemoveNextDocument();
  const { deleteDocument } = useDeleteDocument();
  const { data: documentInfos, setDocumentIds } = useFetchDocumentInfosByIds();
  const { uploadAndParseDocument } = useUploadAndParseDocument(uploadMethod);
  const conversationIdRef = useRef(conversationId);

  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const optionsZh = [
    '自由对话',
    '程序开发',
    '知识库',
    '日志分析',
    'CMDB',
    '交换机'
    // '图片理解',
    // '图片生成'
  ]; // 选择框的选项
  const optionsZh_Tr = [
    '自由對話',
    '程序開發',
    '知識庫',
    '日誌分析',
    'CMDB',
    '交换机'
    // '圖片理解',
    // '圖片生成'
  ]; // 选择框的选项
  const optionsEn = [
    'CHAT',
    'CODE',
    'KNOWLEDGE',
    'LOG',
    'CMDB',
    'SWITCH'

    // 'Image2Txt',
    // 'Txt2Image'
  ]; // 选择框的选项
 
  const optionsMap = {
    'zh': optionsZh,
    'en': optionsEn,
    'zh-TRADITIONAL': optionsZh_Tr
  };
  const optionTipsMap = {
    'zh': '随时@小吉, 使用各种能力...',
    'en': 'Call @ IntelliOps at any time, using  skills...',
    'zh-TRADITIONAL': '隨時@小吉，使用各種能力...'
  };
  const [popoverVisible, setPopoverVisible] = useState(false); // 控制 Popover 是否可见
  const [inputValue, setInputValue] = useState(value); // 输入框的值
  const [placeholderValue, setPlaceholderValue] = useState( t('')); // 输入框的值

  const handlePreview = async (file: UploadFile) => {
    if (!file.url && !file.preview) {
      file.preview = await getBase64(file.originFileObj as FileType);
    }
  };
// 处理输入框内容改变
const handleChangeInput = (e: ChangeEvent<HTMLTextAreaElement>) => {
  const newValue = e.target.value;
  setInputValue(newValue);

  // 如果输入框的值以 "@" 开头，显示 Popover
  if (newValue.startsWith('@')) {
    setPopoverVisible(true);
  } else {
    setPopoverVisible(false);
  }
  if (onInputChange) {
    if (newValue.startsWith('@')){
      e.target.value=''
      onInputChange(e); // 调用传递的输入改变处理函数
    }else{
      onInputChange(e); // 调用传递的输入改变处理函数
    }
    
  }
};

// 当选择一个选项时
const handleSelect = (option: ChangeEvent<HTMLTextAreaElement>) => {
  const newValue = inputValue.replace(/@\S*$/, `@${option} `); // 替换 "@xxxx" 为 "@选中的选项"
  // setInputValue(newValue);
  // setPopoverVisible(false);
  if (onSelect) {
    onSelect(option); // 调用 onSelect 回调，将选定的值传递给父组件
  }

  // 使用 Modal 进行提示
  // 显示成功通知
  notification.success({
    message:  t('msg')+` : ${option} `,
    placement: 'topRight', // 可以设置通知的位置，比如 'topRight', 'bottomLeft' 等
  });
  setPopoverVisible(false);

  // 清空输入框的值
  setInputValue('');
  setPlaceholderValue(`@${option}`);
};

// const { runDocumentByIds } = useRunNextDocument();
// const runDocument = useCallback(
//   (run: number,doc_ids: any) => {
//     runDocumentByIds({
//       documentIds: doc_ids,
//       run,
//     });
//   },
//   [runDocumentByIds],
// );
  const handleChange: UploadProps['onChange'] = async ({
    // fileList: newFileList,
    file,
  }) => {
    let nextConversationId: string = conversationId;
    if (createConversationBeforeUploadDocument) {
      const creatingRet = await createConversationBeforeUploadDocument(
        file.name,
      );
      if (creatingRet?.code === 0) {
        nextConversationId = creatingRet.data.id;
      }
    }
    setFileList((list) => {
      list.push({
        ...file,
        status: 'uploading',
        originFileObj: file as any,
      });
      return [...list];
    });
    const skill=placeholderValue.replace("@","")
    const ret = await uploadAndParseDocument({
      skill: skill,
      conversationId: nextConversationId,
      fileList: [file],
    });
    setFileList((list) => {
      const nextList = list.filter((x) => x.uid !== file.uid);
      nextList.push({
        ...file,
        originFileObj: file as any,
        response: ret,
        percent: 100,
        status: ret?.code === 0 ? 'done' : 'error',
      });
      return nextList;
    });
  };
  // 同步外部的 value 到内部的 inputValue
  useEffect(() => {
    setInputValue(value);
}, [value]);

  const isUploadingFile = fileList.some((x) => x.status === 'uploading');

  const handlePressEnter = useCallback(async () => {
    if (isUploadingFile) return;
    const ids = getFileIds(fileList.filter((x) => isUploadSuccess(x)));    console.log(placeholderValue)
    if(placeholderValue=="@LOG"||placeholderValue=="@日志分析"||placeholderValue=="@日誌分析"){
      if(ids.length<=0){
        notification.error({
          message:  t('uploadfiletips'),
          placement: 'topRight', // 可以设置通知的位置，比如 'topRight', 'bottomLeft' 等
        });
        return
      }
    }
    onPressEnter(ids);
    setFileList([]);
  }, [fileList, onPressEnter, isUploadingFile]);

  const handleKeyDown = useCallback(
    async (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // check if it was shift + enter
      if (event.key === 'Enter' && event.shiftKey) return;
      if (event.key !== 'Enter') return;
      if (sendDisabled || isUploadingFile || sendLoading) return;

      event.preventDefault();
      handlePressEnter();
    },
    [sendDisabled, isUploadingFile, sendLoading, handlePressEnter],
  );

  const handleRemove = useCallback(
    async (file: UploadFile) => {
      const ids = get(file, 'response.data', []);
      // Upload Successfully
      if (Array.isArray(ids) && ids.length) {
        if (isShared) {
          await deleteDocument(ids);
        } else {
          await removeDocument(ids[0]);
        }
        setFileList((preList) => {
          return preList.filter((x) => getFileId(x) !== ids[0]);
        });
      } else {
        // Upload failed
        setFileList((preList) => {
          return preList.filter((x) => x.uid !== file.uid);
        });
      }
    },
    [removeDocument, deleteDocument, isShared],
  );

  const handleStopOutputMessage = useCallback(() => {
    stopOutputMessage?.();
  }, [stopOutputMessage]);

  const getDocumentInfoById = useCallback(
    (id: string) => {
      return documentInfos.find((x) => x.id === id);
    },
    [documentInfos],
  );

  useEffect(() => {
    const ids = getFileIds(fileList);
    setDocumentIds(ids);
  }, [fileList, setDocumentIds]);

  useEffect(() => {
    if (
      conversationIdRef.current &&
      conversationId !== conversationIdRef.current
    ) {
      setFileList([]);
    }
    conversationIdRef.current = conversationId;
  }, [conversationId, setFileList]);

  return (
    <Flex
      gap={1}
      vertical
      className={cn(styles.messageInputWrapper, 'dark:bg-black')}
    >
      <Popover
        content={
          <List
            dataSource={ optionsMap[i18n.language]}
            renderItem={(item: ChangeEvent<HTMLInputElement>) => (
              <List.Item
                onClick={() => handleSelect(item)}
                style={{ cursor: 'pointer' }}
              >
                {item}
              </List.Item>
            )}
          />
        }
        title= {t('selectSkillHolder')}
        trigger="click"
        open={popoverVisible}
        onOpenChange={setPopoverVisible}
        placement="topLeft"
      ></Popover>
      <TextArea
        size="large"
        placeholder={ placeholderValue =='chat.' ? (optionTipsMap[i18n.language]):placeholderValue }
        value={value}
        allowClear
        disabled={disabled}
        style={{
          border: 'none',
          boxShadow: 'none',
          padding: '0px 10px',
          marginTop: 10,
        }}
        autoSize={{ minRows: 2, maxRows: 10 }}
        onKeyDown={handleKeyDown}
        onChange={handleChangeInput}
        onFocus={() => setPopoverVisible(false)} 
      />
      <Divider style={{ margin: '5px 30px 10px 0px' }} />
      <Flex justify="space-between" align="center">
        {fileList.length > 0 && (
          <List
            grid={{
              gutter: 16,
              xs: 1,
              sm: 1,
              md: 1,
              lg: 1,
              xl: 2,
              xxl: 4,
            }}
            dataSource={fileList}
            className={styles.listWrapper}
            renderItem={(item) => {
              const id = getFileId(item);
              const documentInfo = getDocumentInfoById(id);
              const fileExtension = getExtension(documentInfo?.name ?? '');
              const fileName = item.originFileObj?.name ?? '';

              return (
                <List.Item>
                  <Card className={styles.documentCard}>
                    <Flex gap={10} align="center">
                      {item.status === 'uploading' ? (
                        <Spin
                          indicator={
                            <LoadingOutlined style={{ fontSize: 24 }} spin />
                          }
                        />
                      ) : item.status === 'error' ? (
                        <InfoCircleOutlined size={30}></InfoCircleOutlined>
                      ) : (
                        <FileIcon id={id} name={fileName}></FileIcon>
                      )}
                      <Flex vertical style={{ width: '90%' }}>
                        <Text
                          ellipsis={{ tooltip: fileName }}
                          className={styles.nameText}
                        >
                          <b> {fileName}</b>
                        </Text>
                        {item.status === 'error' ? (
                          t('uploadFailed')
                        ) : (
                          <>
                            {item.percent !== 100 ? (
                              t('uploading')
                            ) : !item.response ? (
                              t('parsing')
                            ) : (
                              <Space>
                                <span>{fileExtension?.toUpperCase()},</span>
                                <span>
                                  {formatBytes(
                                    getDocumentInfoById(id)?.size ?? 0,
                                  )}
                                </span>
                              </Space>
                            )}
                          </>
                        )}
                      </Flex>
                    </Flex>

                    {item.status !== 'uploading' && (
                      <span className={styles.deleteIcon}>
                        <CloseCircleOutlined
                          onClick={() => handleRemove(item)}
                        />
                      </span>
                    )}
                  </Card>
                </List.Item>
              );
            }}
          />
        )}
        <Flex
          gap={5}
          align="center"
          justify="flex-end"
          style={{
            paddingRight: 10,
            paddingBottom: 10,
            width: fileList.length > 0 ? '50%' : '100%',
          }}
        >
          {showUploadIcon && (
            <Upload
              onPreview={handlePreview}
              onChange={handleChange}
              multiple={false}
              onRemove={handleRemove}
              showUploadList={false}
              beforeUpload={() => {
                return false;
              }}
            >
              <Button type={'primary'} disabled={disabled}>
                <Paperclip className="size-4" />
              </Button>
            </Upload>
          )}
          {sendLoading ? (
            <Button onClick={handleStopOutputMessage}>
              <CircleStop className="size-5" />
            </Button>
          ) : (
            <Button
              type="primary"
              onClick={handlePressEnter}
              loading={sendLoading}
              disabled={sendDisabled || isUploadingFile || sendLoading}
            >
              <SendHorizontal className="size-5" />
            </Button>
          )}
        </Flex>
      </Flex>
    </Flex>
  );
};

export default memo(MessageInput);
